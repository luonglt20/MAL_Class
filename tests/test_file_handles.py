import copy
import json

import pytest
import torch
from unittest.mock import patch

from malware_hybrid.dataset import ReportTensorizer, collate_samples
from malware_hybrid.graph import ExecutionGraphBuilder
from malware_hybrid.handles import FileHandleResolver
from malware_hybrid.model import TemporalHierarchicalEncoder
from malware_hybrid.normalization import CapeReportParser, CAPE_FEATURE_SCHEMA_VERSION, FILE_HANDLE_SCHEMA_VERSION


def call(api, args, success=True, returned=0):
    return {"api": api, "category": "file", "status": success, "return": returned, "arguments": args}


def parse(calls, owner=100, enabled=True):
    return CapeReportParser(file_handle_resolution=enabled, collapse_repeats=False).parse({
        "behavior": {"processes": [{"process_id": owner, "calls": calls}]},
    })


def opened(path=r"C:\Users\alice\payload.exe", handle=68):
    return call("NtCreateFile", {"FileName": path, "FileHandle": handle})


def test_open_read_write_share_anonymous_resource_and_graph_node():
    report = parse([opened(), call("NtWriteFile", {"FileHandle": "0x44"}),
                    call("NtReadFile", {"FileHandle": 68})])
    assert len({event.target for event in report.events}) == 1
    assert report.events[0].target.startswith("entity:")
    assert [event.target_resolution for event in report.events] == [
        "file_handle_acquisition", "file_handle", "file_handle"]
    graph = ExecutionGraphBuilder().build(report.events)
    assert len([node for node in graph.nodes if node.node_type == "file"]) == 1
    assert "alice" not in json.dumps([event.public_dict() for event in report.events])
    assert report.metadata["normalization_version"] == FILE_HANDLE_SCHEMA_VERSION


def test_default_schema_and_historical_outputs_stay_v3():
    calls = [opened(), call("NtReadFile", {"FileHandle": 68})]
    legacy = parse(copy.deepcopy(calls), enabled=False)
    assert legacy.metadata["normalization_version"] == CAPE_FEATURE_SCHEMA_VERSION
    assert legacy.events[1].target == "<FILE_HANDLE_UNKNOWN>"
    assert legacy.events[0].target != legacy.events[1].target


def test_resolved_file_identity_reaches_temporal_attention_relation_bias():
    report = parse([opened(), call("NtReadFile", {"FileHandle": 68}),
                    call("NtWriteFile", {"FileHandle": 68})])
    tensorizer = ReportTensorizer.fit([report])
    sample = tensorizer.encode(report)
    assert len(set(sample.resource_ids)) == 1 and sample.resource_ids[0] > 0
    batch = collate_samples([sample])
    encoder = TemporalHierarchicalEncoder(
        len(tensorizer.token_vocab), 8, 2, 1, 0., 4, 2, 32, 32,
    ).eval()

    def run(resources):
        with patch.object(encoder.event_blocks[0], "forward", wraps=encoder.event_blocks[0].forward) as block:
            encoder(*(batch[key] for key in (
                "event_ids", "event_times", "event_success", "event_repeats", "process_ids",
                "parent_process_ids", "thread_ids", "event_mask")), resources, return_attention=False)
        return block.call_args.args[2].clone()

    with torch.inference_mode():
        linked = run(batch["resource_ids"])
        unknown = run(torch.zeros_like(batch["resource_ids"]))
    # Input to the attention logits changes through the resource relation,
    # rather than treating a numeric handle as an API-vocabulary feature.
    torch.testing.assert_close(linked[0, :, 0, 1] - unknown[0, :, 0, 1],
                               encoder.relation_weights[:, 3])


@pytest.mark.parametrize("api,args", [("NtClose", {"Handle": 68}),
                                      ("ZwClose", {"Handle": "0x44"}),
                                      ("CloseHandle", {"hObject": 68})])
def test_successful_close_removes_binding_and_failed_close_keeps_it(api, args):
    report = parse([opened(), call(api, args, success=False), call("NtReadFile", {"FileHandle": 68}),
                    call(api, args), call("NtReadFile", {"FileHandle": 68})])
    assert report.events[1].target == report.events[0].target
    assert not report.events[1].success
    assert report.events[2].target == report.events[0].target
    assert report.events[3].target_resolution == "closed_file_handle"
    assert report.events[4].target == "<FILE_HANDLE_UNKNOWN>"


def test_reuse_and_conflicting_explicit_name_never_link_wrong_file():
    report = parse([opened(), opened(r"C:\Users\alice\other.exe"),
                    call("NtReadFile", {"FileHandle": 68}),
                    call("NtReadFile", {"FileHandle": 68, "HandleName": r"C:\elsewhere\payload.exe"}),
                    call("NtWriteFile", {"FileHandle": 68})])
    assert report.events[0].target != report.events[1].target
    assert report.events[2].target == report.events[1].target
    assert report.events[3].target_resolution == "conflicting_file_handle"
    assert report.events[4].target == "<FILE_HANDLE_UNKNOWN>"


@pytest.mark.parametrize("path", ["payload.exe", r"folder\payload.exe", "0x44", "<FILE>", "C:\\bad\x00.exe"])
def test_relative_opaque_or_nul_names_do_not_establish_bindings(path):
    report = parse([opened(path), call("NtReadFile", {"FileHandle": 68})])
    assert all(event.target == "<FILE_HANDLE_UNKNOWN>" for event in report.events)


def test_failed_open_does_not_bind_and_failed_use_is_not_success_evidence():
    first = opened(); first["status"] = False
    report = parse([first, call("NtReadFile", {"FileHandle": 68}), opened(),
                    call("NtWriteFile", {"FileHandle": 68}, success=False)])
    assert report.events[1].target == "<FILE_HANDLE_UNKNOWN>"
    assert report.events[3].target == report.events[2].target
    assert not report.events[3].success


def test_handle_name_can_describe_a_typed_file_without_inventing_acquisition():
    report = parse([call("NtReadFile", {"FileHandle": 68, "HandleName": r"C:\Temp\x.dll"}),
                    call("NtWriteFile", {"FileHandle": 68})])
    assert report.events[0].target_resolution == "file_path"
    assert report.events[1].target == "<FILE_HANDLE_UNKNOWN>"


@pytest.mark.parametrize("owner", ["unknown", "unknown:1", 0])
def test_unknown_owner_never_establishes_handle_binding(owner):
    report = parse([opened(), call("NtReadFile", {"FileHandle": 68})], owner=owner)
    assert report.events[1].target == "<FILE_HANDLE_UNKNOWN>"


def test_same_numeric_handle_is_not_shared_by_processes_or_unknown_apis():
    resolver = FileHandleResolver()
    assert resolver.resolve("100", "NtCreateFile", {"filehandle": 68}, 0, True,
                            "entity:0:<USER_PATH>", "file")[2] == "file_handle_acquisition"
    for owner, api in [("200", "NtReadFile"), ("100", "UnknownApi")]:
        assert resolver.resolve(owner, api, {"filehandle": 68}, 0, True,
                                "<FILE_HANDLE_UNKNOWN>", "file")[0] == "<FILE_HANDLE_UNKNOWN>"


@pytest.mark.parametrize("acquisition", [
    call("OpenProcess", {}, returned=68),
    call("NtOpenProcess", {"ProcessHandle": 68}),
    call("NtOpenKey", {"KeyHandle": 68}),
    opened("relative.exe"),
])
def test_successful_reacquisition_invalidates_old_file_even_without_new_target(acquisition):
    report = parse([opened(), acquisition, call("NtReadFile", {"FileHandle": 68})])
    assert report.events[-1].target == "<FILE_HANDLE_UNKNOWN>"


def test_win32_return_handle_and_distinct_same_semantic_paths():
    report = parse([call("CreateFileW", {"lpFileName": r"C:\Temp\a.exe"}, returned=68),
                    call("ReadFile", {"hFile": 68}),
                    opened(r"C:\Temp\b.exe", handle=72), call("NtReadFile", {"FileHandle": 72})])
    assert report.events[0].target == report.events[1].target
    assert report.events[2].target == report.events[3].target
    assert report.events[0].target != report.events[2].target


def test_contradictory_names_on_acquisition_do_not_bind():
    event = opened(); event["arguments"]["HandleName"] = r"C:\different\file.exe"
    report = parse([event, call("NtReadFile", {"FileHandle": 68})])
    assert report.events[0].target_resolution == "conflicting_file_names"
    assert report.events[1].target == "<FILE_HANDLE_UNKNOWN>"


@pytest.mark.parametrize("handle", [0, -1, "0xffffffff", "0xffffffffffffffff", True])
def test_invalid_open_handle_is_not_successful_even_if_report_status_says_true(handle):
    report = parse([opened(handle=handle)])
    assert report.events[0].target_resolution == "invalid_file_handle"
    assert not report.events[0].success
