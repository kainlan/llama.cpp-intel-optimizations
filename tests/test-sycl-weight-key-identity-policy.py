#!/usr/bin/env python3
"""Weight cache key = physical identity, source contract (llama.cpp-n3pw, ruling c-0bvm).

Host-only: reads sources, runs no build, loads no model and touches no device.

The executable tests prove the policy against keys they construct --
test-sycl-weight-key-stability (a GGUF-backed key carries its file identity),
test-sycl-weight-key-uniqueness (distinct weights stay distinct, tied weights
share), test-sycl-moe-identity-hash (model separation comes from the file, not
from a model_id field). What none of them can see is WHY a key came out right,
and that is what erodes: the fields excluded from equality are excluded by the
absence of a line, and an absence is restored by anyone who reads the exclusion
as an oversight. Two of the three would still pass with model_id or name_hash
quietly folded back into the GGUF-backed comparison -- their tensors differ in
file identity too.

So this gate holds the shape of the decision:

  * cache_id_equal and cache_id_hash treat a GGUF-backed weight as its bytes --
    (file_id, file_offs, nbytes, type, ne) -- with model_id and name_hash
    consulted only for weights that have no file identity, and the hash
    excluding exactly what equality excludes;
  * file_id exists and is populated, because file_idx alone is a split index
    within one model and every model has a file_idx 0 -- dropping model_id from
    the comparison is safe only against a WHOLE-FILE identity;
  * an identity registered with no load transaction open must declare its model
    and is readable only while no model is loaded, so the host-testable path
    cannot become a way for two models to share a key by accident;
  * the loader publishes each split's identity before the tensors that live in
    it.

Every check below fails against the pre-n3pw tree. Run it that way to confirm:

    ./tests/test-sycl-weight-key-identity-policy.py \\
        --key <(git show f46129c5b:ggml/src/ggml-sycl/unified-cache-key.hpp) \\
        --sycl <(git show f46129c5b:ggml/src/ggml-sycl/ggml-sycl.cpp) \\
        --header <(git show f46129c5b:ggml/include/ggml-sycl.h) \\
        --loader <(git show f46129c5b:src/llama-model-loader.cpp)

The `anchors` section proves every region the checks read was actually found, so
a renamed function reports a missing anchor instead of passing vacuously. Run
with --self-test to prove the absence-based checks fire against mutants.
"""
import argparse
import re
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser()
parser.add_argument("--key", default=str(root / "ggml/src/ggml-sycl/unified-cache-key.hpp"))
parser.add_argument("--sycl", default=str(root / "ggml/src/ggml-sycl/ggml-sycl.cpp"))
parser.add_argument("--header", default=str(root / "ggml/include/ggml-sycl.h"))
parser.add_argument("--loader", default=str(root / "src/llama-model-loader.cpp"))
parser.add_argument("--self-test", action="store_true",
                    help="prove the absence-based checks fire when the thing they forbid is present")
args = parser.parse_args()


def strip_comments(source):
    """Remove C and C++ comments, preserving string literals and line count.

    Every check below reads active code only. Without this, "model_id is not
    compared for a GGUF-backed weight" is spoofed by the comment that explains
    why it is not -- and this change writes exactly that comment.
    """
    out = []
    i = 0
    n = len(source)
    while i < n:
        ch = source[i]
        if ch in ('"', "'"):
            quote = ch
            out.append(ch)
            i += 1
            while i < n:
                out.append(source[i])
                if source[i] == "\\":
                    if i + 1 < n:
                        out.append(source[i + 1])
                        i += 2
                        continue
                elif source[i] == quote:
                    i += 1
                    break
                i += 1
            continue
        if source.startswith("//", i):
            while i < n and source[i] != "\n":
                i += 1
            continue
        if source.startswith("/*", i):
            end = source.find("*/", i + 2)
            end = n if end < 0 else end + 2
            out.append("\n" * source.count("\n", i, end))
            i = end
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def squeeze(source):
    """Collapse runs of spaces/tabs so the repo's vertical alignment is invisible."""
    return re.sub(r"[ \t]+", " ", source)


def between(source, start, end):
    """Text from the first `start` to the first `end` that follows it."""
    at = source.find(start)
    if at < 0:
        return ""
    stop = source.find(end, at + len(start))
    return source[at:stop + len(end)] if stop >= 0 else ""


def ordered(text, *needles):
    """True when every needle appears, in the given order."""
    at = -1
    for needle in needles:
        found = text.find(needle, at + 1)
        if found < 0:
            return False
        at = found
    return True


def evaluate(key, sycl, header, loader):
    key, sycl = squeeze(key), squeeze(sycl)
    header, loader = squeeze(header), squeeze(loader)

    tripwire = between(key, "static_assert(sizeof(ggml_sycl_cache_id)", ");")
    equal = between(key, "static inline bool cache_id_equal(", "struct cache_id_equal_fn")
    hashfn = between(key, "struct cache_id_hash {", "\n};")
    # The comparison that runs for EVERY key, GGUF-backed or not. What appears
    # here is what a GGUF-backed weight is identified by.
    equal_physical = between(equal, "if (a.file_id != b.file_id", "return true;")
    # Likewise for the hash: everything after the has_gguf-guarded block.
    hash_physical = between(hashfn, "std::hash<uint64_t>()(id.file_id)", "return h;")

    cache_id_struct = between(header, "struct ggml_sycl_cache_id {", "};")
    register_weight = between(sycl, "void ggml_backend_sycl_register_weight_identity(",
                              "static tensor_usage ggml_sycl_usage_from_api(")
    register_file = between(sycl, "void ggml_backend_sycl_register_gguf_file_identity(",
                            "void ggml_backend_sycl_register_weight_identity(")
    lookup = between(sycl, "ggml_sycl_cache_id ggml_backend_sycl_get_weight_cache_key(",
                     "ggml_sycl_cache_id ggml_backend_sycl_get_tensor_cache_key(")
    # The identity resolution inside the lookup: owner first, unowned table only
    # when there is no owner at all.
    resolve = between(lookup, "if (!name.empty() && name != \"unknown\") {", "if (has_gguf_identity) {")
    erase = between(sycl, "static void ggml_sycl_erase_weight_identities_for_owner(",
                    "static void ggml_sycl_release_model_slot_resources(")
    loader_register = between(loader, "auto register_sycl_tensor_metadata = ", "auto usage_from_tensor = ")

    anchors = {
        "ggml_sycl_cache_id size tripwire": tripwire,
        "cache_id_equal body": equal,
        "cache_id_hash body": hashfn,
        "cache_id_equal physical comparison": equal_physical,
        "cache_id_hash physical mixing": hash_physical,
        "ggml_sycl_cache_id declaration": cache_id_struct,
        "ggml_backend_sycl_register_weight_identity body": register_weight,
        "ggml_backend_sycl_register_gguf_file_identity body": register_file,
        "ggml_backend_sycl_get_weight_cache_key body": lookup,
        "weight cache key identity resolution": resolve,
        "ggml_sycl_erase_weight_identities_for_owner body": erase,
        "loader register_sycl_tensor_metadata body": loader_register,
    }

    checks = {
        # The field list is written out by hand in four places and nothing in the
        # language ties them together: a field added to three of the four makes
        # two different weights compare equal, silently. The tripwire is the only
        # thing that turns that into a build failure, and its message is the only
        # thing that tells whoever hits it where the other three sites are -- so
        # the message must still name all four.
        "a size tripwire guards every copy of the field list": (
            tripwire != ""
            and all(site in tripwire for site in ("cache_id_equal", "cache_id_hash", "same_logical_moe_expert",
                                                  "retained_cache_id_less"))),

        # The whole ruling in one line: for a GGUF-backed weight, which model
        # loaded it and what a graph node called it are not part of what it is.
        # Both must be reachable only under the !has_gguf guard, or a tied
        # embedding/output head splits into two staged copies again.
        "equality consults model_id and name_hash only without file identity": (
            "const bool compare_logical = !a.has_gguf;" in equal
            and "if (compare_logical && (a.model_id != b.model_id || a.name_hash != b.name_hash)) {" in equal
            and "model_id" not in equal_physical
            and "name_hash" not in equal_physical),

        # A hash that mixes a field equality ignores puts two equal keys in
        # different buckets, so the map never finds the entry it already holds.
        # This is the failure that looks like a cache miss, not like a bug.
        "the hash excludes exactly what equality excludes": (
            "if (!id.has_gguf) {" in hashfn
            and "std::hash<uint64_t>()(id.model_id)" in hashfn
            and "std::hash<uint64_t>()(id.name_hash)" in hashfn
            and "model_id" not in hash_physical
            and "name_hash" not in hash_physical),

        # file_idx is a split index WITHIN one model. Comparing it in place of a
        # file identity means every model's split 0 is the same file, which is
        # precisely the collision dropping model_id would otherwise open.
        "the key carries a whole-file identity, not just a split index": (
            "uint64_t file_id;" in cache_id_struct
            and "uint16_t file_idx;" in cache_id_struct
            and "a.file_id != b.file_id" in equal_physical
            and "std::hash<uint64_t>()(id.file_id)" in hashfn),

        "a GGUF-backed key is populated from the registered identity": (
            "id.file_id = has_gguf_identity ? identity.file_id : 0;" in lookup
            and "id.file_idx = has_gguf_identity ? identity.file_idx : 0;" in lookup
            and "id.file_offs = has_gguf_identity ? identity.file_offs : 0;" in lookup),

        # An unpublished split still gets an identity, derived from the model.
        # That forfeits sharing between two models over one file; it must never
        # forfeit the isolation, so the model id has to be mixed in.
        "an unpublished split falls back to a per-model file identity": (
            "static uint64_t ggml_sycl_file_id_from_model(uint64_t model_id, uint16_t file_idx) {" in sycl
            and ordered(between(sycl, "static uint64_t ggml_sycl_file_id_from_model(", "\n}"),
                        "std::hash<uint64_t>()(model_id)", "std::hash<uint16_t>()(file_idx)")
            and "file_id = ggml_sycl_file_id_from_model(model_id, file_idx);" in register_weight),

        # Outside a load transaction there is no lifecycle owner to attribute a
        # registration to, so the caller is the only source of model identity.
        # Accepting model_id == 0 there would file a weight under no model at
        # all -- and its file identity falls back to that same model id.
        "an unowned registration must declare its model": ordered(
            register_weight,
            "if (!effect) {",
            "if (model_id == 0) {",
            "return;",
            "identity.file_id = ggml_sycl_file_id_from_model(model_id, file_idx);",
            "g_sycl_weight_identities_unowned[name] = identity;"),

        # The unowned table is keyed by bare tensor name, so it is safe only
        # where no model is loaded to collide with. Reading it whenever the
        # owner-scoped lookup misses would hand a live model another model's
        # file offsets under a matching name.
        "unowned identities are read only when nothing is loaded": ordered(
            resolve,
            "if (owner.model.value != 0) {",
            "g_sycl_weight_identities_by_name.find(",
            "} else {",
            "g_sycl_weight_identities_unowned.find(name)")
            # Ordering alone would still admit a second, unguarded read placed
            # ahead of the owner lookup, which is the whole hazard.
            and resolve.find("g_sycl_weight_identities_unowned") > resolve.find("} else {"),

        # With two models loaded the published plan names one of them; the other
        # would find no identity and drop to the fallback UUID path, losing the
        # file identity this whole change is about.
        "the lookup resolves the owner from the tensor before the published plan": ordered(
            resolve,
            "if (extra && extra->model_id != 0) {",
            "ggml_sycl::lifecycle::global_registry().find({ extra->model_id })",
            "if (owner.model.value == 0) {",
            "owner = ggml_sycl_identity_owner(ggml_sycl_identity_plan_snapshot());"),

        # Published split identities carry the owner prefix and must die with the
        # owner, or a later load reusing the slot inherits this one's files.
        "released model slots drop their published split identities": (
            "g_sycl_gguf_file_ids.erase(it)" in erase
            and "ggml_sycl_owner_name_key_matches(it->first, owner)" in erase),

        # A split identity registered after its tensors is registered too late:
        # those weights already fell back to the model-derived identity, so two
        # models over one file silently stop sharing.
        "the loader publishes a split's identity before its tensors": ordered(
            loader_register,
            "sycl_hooks.register_file_identity(weight->second.idx, file_paths[idx].c_str(),",
            "sycl_hooks.register_identity(tensor, weight->second.idx, weight->second.offs,"),

        # An empty path hashes to nothing distinguishing, so publishing one would
        # make every pathless model claim the same file.
        "the loader publishes nothing for a split it has no path for": (
            "!file_paths[idx].empty()" in loader_register
            and "if (path == nullptr || path[0] == '\\0') {" in
                between(sycl, "static uint64_t ggml_sycl_file_id_from_path(", "\n}")),
    }

    missing = [name for name, text in anchors.items() if not text]
    failed = [name for name, ok in checks.items() if not ok]
    return missing, failed, len(checks)


# Each entry: (which source, anchor to replace, replacement that reintroduces
# the forbidden construct). Absence checks pass by default, so each one needs a
# mutant that proves it can fail at all.
ABSENCE_MUTANTS = {
    # The obvious "restoration": model_id looks like it belongs in an identity,
    # and putting it back is a one-line edit that no runtime test here catches.
    "equality consults model_id and name_hash only without file identity": (
        "key",
        "a.nbytes != b.nbytes ||",
        "a.nbytes != b.nbytes || a.model_id != b.model_id ||"),
    # Hash and equality drift apart silently: nothing crashes, entries just stop
    # being found. Mixing name_hash unconditionally is the natural way to do it.
    "the hash excludes exactly what equality excludes": (
        "key",
        "h = cache_hash_combine(h, std::hash<uint64_t>()(id.file_id));",
        "h = cache_hash_combine(h, std::hash<uint64_t>()(id.file_id));\n"
        " h = cache_hash_combine(h, std::hash<uint64_t>()(id.name_hash));"),
    # Dropping the declaration check turns the unowned path into "register under
    # model 0", which is every unowned caller sharing one identity space.
    "an unowned registration must declare its model": (
        "sycl",
        "if (model_id == 0) {\n return;\n }\n ggml_sycl::dispatch_tuning::ensure_model_loaded(model_id);",
        "ggml_sycl::dispatch_tuning::ensure_model_loaded(model_id);"),
    # Reading the unowned table on any miss is the change that makes the
    # host-testable path reachable from a live model.
    "unowned identities are read only when nothing is loaded": (
        "sycl",
        "if (owner.model.value != 0) {\n auto name_it = g_sycl_weight_identities_by_name.find(",
        "auto unowned_probe = g_sycl_weight_identities_unowned.find(name);\n"
        " if (owner.model.value != 0) {\n auto name_it = g_sycl_weight_identities_by_name.find("),
    # Publishing a fabricated identity for a pathless split makes every such
    # model claim the same file -- the exact cross-model aliasing this closes.
    "the loader publishes nothing for a split it has no path for": (
        "loader",
        "idx < file_paths.size() &&\n !file_paths[idx].empty()",
        "idx < file_paths.size()"),
}


def self_test(key, sycl, header, loader):
    """Every absence check must fail once its forbidden construct is injected."""
    problems = []
    for name, (target, anchor, replacement) in ABSENCE_MUTANTS.items():
        sources = {"key": key, "sycl": sycl, "header": header, "loader": loader}
        anchor, replacement = squeeze(anchor), squeeze(replacement)
        if anchor not in squeeze(sources[target]):
            problems.append("mutation anchor missing for: " + name)
            continue
        sources[target] = squeeze(sources[target]).replace(anchor, replacement, 1)
        _, failed, _ = evaluate(sources["key"], sources["sycl"], sources["header"], sources["loader"])
        if name not in failed:
            problems.append("check did not fire on its mutant: " + name)
    return problems


def read(path):
    return strip_comments(Path(path).read_text()) if Path(path).exists() else ""


key, sycl = read(args.key), read(args.sycl)
header, loader = read(args.header), read(args.loader)

missing_anchors, failed, n_checks = evaluate(key, sycl, header, loader)

for name in missing_anchors:
    print("MISSING ANCHOR: " + name)
for name in failed:
    print("FAIL: " + name)

if missing_anchors or failed:
    sys.exit(1)

if args.self_test:
    problems = self_test(key, sycl, header, loader)
    for problem in problems:
        print("SELF-TEST FAIL: " + problem)
    if problems:
        sys.exit(1)
    print("sycl weight key identity policy: self-test PASS ({} mutants)".format(len(ABSENCE_MUTANTS)))

print("sycl weight key identity policy: PASS ({} checks)".format(n_checks))
