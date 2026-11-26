# Third-party material

The source code in this repository is MIT (see [LICENSE](LICENSE)). The items
below are not, and the MIT grant does not extend to them.

## Depth Pro 2 (Apple)

- **Weights**: `apple/DepthPro2-H` (872 M parameters) and `apple/DepthPro2-focal`,
  downloaded at runtime into `torch_cache/`.
- **Package**: <https://github.com/apple-aiml-research/ml-depth-pro-2>, installed
  from source by the setup steps in the README.

Both are released under the
[Apple Machine Learning Research Model Licence](https://github.com/apple-aiml-research/ml-depth-pro-2/blob/main/LICENSE_MODEL),
which permits **non-commercial scientific research and academic development
only**. Nothing here grants rights to the weights or to that package, and
running this app commercially would need a separate licence from Apple.

## Sample image

`static/sample.jpg` — and the renders derived from it in `docs/` — is Apple's
example photo from the ml-depth-pro-2 repository. It is included only so the
"Try a sample" button has something to show. Treat it as research-use material
and replace it before redistributing this project for other purposes.
