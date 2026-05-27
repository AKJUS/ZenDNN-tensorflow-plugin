# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ******************************************************************************

"""zentf: A TensorFlow extension for AMD EPYC CPUs."""

import warnings
from importlib import metadata as _metadata


def _check_dual_install():
  """Raise ImportError if both zentf and zentf-weekly are installed."""
  installed_dists = []
  for _name in ("zentf", "zentf-weekly"):
    try:
      _metadata.version(_name)
      installed_dists.append(_name)
    except _metadata.PackageNotFoundError:
      pass
  if len(installed_dists) > 1:
    raise ImportError(
        f"Both {' and '.join(installed_dists)} are installed. "
        "Please uninstall one of them, for example:\n"
        "  pip uninstall zentf\n"
        "  pip uninstall zentf-weekly"
    )


_check_dual_install()

# Import build info generated at wheel build time
try:
  from ._build_info import (
      __zentf_commit__, __zendnn_version__,
      __tf_version__, __source_tag__, __release_type__,
  )
except ImportError:
  __zentf_commit__ = "unknown"
  __zendnn_version__ = "unknown"
  __tf_version__ = "unknown"
  __source_tag__ = ""
  __release_type__ = "unknown"


def _check_tf_compatibility():
  """Warn if the installed TensorFlow major.minor differs from build-time TF."""
  if __tf_version__ == "unknown":
    return
  try:
    import tensorflow as tf
    installed_ver = tf.__version__
  except ImportError:
    raise ImportError(
        "TensorFlow is not installed. zentf requires TensorFlow to function.\n"
        f"This zentf package was built with TensorFlow {__tf_version__}.\n"
        f"  pip install tensorflow=={__tf_version__}"
    ) from None

  build_parts = __tf_version__.split('.')[:2]
  installed_parts = installed_ver.split('.')[:2]
  if build_parts != installed_parts:
    warnings.warn(
        f"zentf was built with TensorFlow {__tf_version__} but TensorFlow "
        f"{installed_ver} is installed. This may cause compatibility issues. "
        f"Please install a matching TensorFlow version:\n"
        f"  pip install tensorflow=={__tf_version__}",
        RuntimeWarning,
        stacklevel=2,
    )


_check_tf_compatibility()

# Get package version from metadata (handles both GA and weekly package names)
def _get_version():
  for name in ("zentf", "zentf-weekly"):
    try:
      return _metadata.version(name)
    except _metadata.PackageNotFoundError:
      continue
  return "unknown"


__version__ = _get_version()


def show_config():
  """Return a string describing the zentf build configuration.

  Returns:
    str: A formatted string containing version and build information.
  """
  # Determine if zendnn_version is a commit hash or version tag
  if __zendnn_version__ and len(__zendnn_version__) <= 12 and all(c in \
    '0123456789abcdef' for c in __zendnn_version__):
    zendnn_label = f"AMD ZenDNNL ( Git Hash {__zendnn_version__} )"
  else:
    ver = __zendnn_version__.lstrip('v') \
        if __zendnn_version__ else __zendnn_version__
    zendnn_label = f"AMD ZenDNNL v{ver}"

  config = f"""zentf Version: {__version__}
Release Type: {__release_type__}
zentf built with:
  - Source Tag: {__source_tag__ or 'untagged'}
  - Commit-id: {__zentf_commit__}
  - TensorFlow: {__tf_version__}
Third_party libraries:
  - {zendnn_label}
"""
  return config


__config__ = show_config()
