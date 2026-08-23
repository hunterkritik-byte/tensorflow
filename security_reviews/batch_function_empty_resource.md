# BatchFunction captured `DT_RESOURCE` safety review

## Finding

During review of `tensorflow/core/kernels/batch_kernels.cc`, the `captured_tensors` path handles `DT_RESOURCE` tensors by indexing element zero:

```cpp
const ResourceHandle& rhandle = t.flat<ResourceHandle>()(0);
```

The path does not first establish that the tensor contains an element.

The corresponding `in_tensors` path explicitly rejects `DT_RESOURCE`, making the captured-resource path a distinct validation surface.

## Proposed mitigation

Validate the element count before indexing:

```cpp
if (t.dtype() == DT_RESOURCE) {
  if (t.NumElements() != 1) {
    return absl::InvalidArgumentError(
        "Captured resource tensors must contain exactly one ResourceHandle");
  }
  const ResourceHandle& rhandle = t.flat<ResourceHandle>()(0);
  opts.input_devices.push_back(rhandle.device());
}
```

## Regression test

Add a test that supplies an empty `DT_RESOURCE` captured tensor and verifies `INVALID_ARGUMENT`. Also test a one-element resource tensor to preserve the valid case.

## Security assessment

An empty tensor reaching the unchecked element access can constitute an out-of-bounds read. Exploitability and severity have **not** been established by this review alone and require reproduction in a supported build.

## Validation status

No ASan/UBSan output is claimed here because the TensorFlow test suite was not executed in this environment. Before merging a source fix, run the targeted kernel tests and sanitizer builds and record the actual output.
