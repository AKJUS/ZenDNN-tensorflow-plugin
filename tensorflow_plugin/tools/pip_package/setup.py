#*******************************************************************************
# Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#*******************************************************************************

import datetime
import fnmatch
import os
import re

import sys

from setuptools import Command
from setuptools import setup
from setuptools.command.install import install as InstallCommandBase
from setuptools.dist import Distribution

# Auto-detect TF version from package metadata (avoids importing TF).
# Only _PLUGIN_PATCH is maintained manually.
_PLUGIN_PATCH = '0'

def _detect_tf_version():
  """Return a clean MAJOR.MINOR.PATCH.PLUGIN_PATCH version string.

  Checks TF_VERSION env var first (set by build_pip_package.sh), then
  probes package metadata for tensorflow, tensorflow-cpu, and tf-nightly.
  Strips pre/post/dev markers (rc, .post, .dev) to produce a valid
  PEP 440 version.
  """
  import re as _re
  raw = os.environ.get('TF_VERSION', '')
  if not raw:
    from importlib.metadata import version as _pkg_version
    for dist_name in ('tensorflow', 'tensorflow-cpu', 'tf-nightly'):
      try:
        raw = _pkg_version(dist_name)
        break
      except Exception:
        continue
  if not raw:
    return '2.21.0.%s' % _PLUGIN_PATCH
  clean = _re.split(r'(\.dev|\.post|rc|-)', raw)[0]
  parts = clean.split('.')[:3]
  while len(parts) < 3:
    parts.append('0')
  return '%s.%s.%s.%s' % (parts[0], parts[1], parts[2], _PLUGIN_PATCH)

_VERSION = _detect_tf_version()
# this path can't be modified.
_PLUGIN_LIB_PATH = 'tensorflow-plugins'
_MY_PLUGIN_PATH = 'zentf'

REQUIRED_PACKAGES = []

if sys.byteorder == 'little':
  # grpcio does not build correctly on big-endian machines due to lack of
  # BoringSSL support.
  # See https://github.com/tensorflow/tensorflow/issues/17882.
  REQUIRED_PACKAGES.append('grpcio >= 1.8.6')

# ZENTF_RELEASE_TYPE: "ga" (default) or "weekly"
_RELEASE_TYPE = os.environ.get('ZENTF_RELEASE_TYPE', 'ga').lower()

if _RELEASE_TYPE == 'weekly':
  project_name = 'zentf-weekly'
  _dev_date = (
      os.environ.get('ZENTF_WEEKLY_DATE', '')
      or datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d')
  )
  if not (_dev_date.isdigit() and len(_dev_date) == 8):
    raise RuntimeError(
        f"Invalid ZENTF_WEEKLY_DATE value: {_dev_date!r}. "
        "It must be an 8-digit date in YYYYMMDD format (e.g., 20250318)."
    )
  _VERSION = _VERSION + '.dev' + _dev_date
else:
  project_name = 'zentf'

# numpy v1.26.4 requires to perform zentf well with TF v2.18.
REQUIRED_PACKAGES.append('numpy == 1.26.4')
# python3 requires wheel 0.26
if sys.version_info.major == 3:
  REQUIRED_PACKAGES.append('wheel >= 0.26')
else:
  REQUIRED_PACKAGES.append('wheel')
  # mock comes with unittest.mock for python3, need to install for python2
  REQUIRED_PACKAGES.append('mock >= 2.0.0')

# weakref.finalize and enum were introduced in Python 3.4
if sys.version_info < (3, 4):
  REQUIRED_PACKAGES.append('backports.weakref >= 1.0rc1')
  REQUIRED_PACKAGES.append('enum34 >= 1.1.6')

CONSOLE_SCRIPTS = []

TEST_PACKAGES = [
    'scipy >= 0.15.1',
]


class BinaryDistribution(Distribution):

  def has_ext_modules(self):
    return True


class InstallCommand(InstallCommandBase):
  """Override the dir where the headers go."""

  def finalize_options(self):
    ret = InstallCommandBase.finalize_options(self) # pylint: disable=assignment-from-no-return
    self.install_headers = os.path.join(self.install_purelib,
                                        'tensorflow-plugins', 'include')
    return ret


class InstallHeaders(Command):
  """Override how headers are copied.

  The install_headers that comes with setuptools copies all files to
  the same directory. But we need the files to be in a specific directory
  hierarchy for -I <include_dir> to work correctly.
  """
  description = 'install C/C++ header files'

  user_options = [('install-dir=', 'd',
                   'directory to install header files to'),
                  ('force', 'f',
                   'force installation (overwrite existing files)'),
                 ]

  boolean_options = ['force']

  def initialize_options(self):
    self.install_dir = None
    self.force = 0
    self.outfiles = []

  def finalize_options(self):
    self.set_undefined_options('install',
                               ('install_headers', 'install_dir'),
                               ('force', 'force'))

  def mkdir_and_copy_file(self, header):
    install_dir = os.path.join(self.install_dir, os.path.dirname(header))
    # Get rid of some extra intervening directories so we can have fewer
    # directories for -I
    install_dir = re.sub('/google/protobuf_archive/src', '', install_dir)

    # Copy external code headers into tensorflow/include.
    # A symlink would do, but the wheel file that gets created ignores
    # symlink within the directory hierarchy.
    # NOTE(keveman): Figure out how to customize bdist_wheel package so
    # we can do the symlink.
    external_header_locations = [
        'tensorflow-plugins/include/external/eigen_archive/',
        'tensorflow-plugins/include/external/com_google_absl/',
        'tensorflow-plugins/include/external/com_google_protobuf',
    ]
    for location in external_header_locations:
      if location in install_dir:
        extra_dir = install_dir.replace(location, '')
        if not os.path.exists(extra_dir):
          self.mkpath(extra_dir)
        self.copy_file(header, extra_dir)

    if not os.path.exists(install_dir):
      self.mkpath(install_dir)
    return self.copy_file(header, install_dir)

  def run(self):
    hdrs = self.distribution.headers
    if not hdrs:
      return

    self.mkpath(self.install_dir)
    for header in hdrs:
      (out, _) = self.mkdir_and_copy_file(header)
      self.outfiles.append(out)

  def get_inputs(self):
    return self.distribution.headers or []

  def get_outputs(self):
    return self.outfiles


def find_files(pattern, root):
  """Return all the files matching pattern below root dir."""
  for dirpath, _, files in os.walk(root):
    for filename in fnmatch.filter(files, pattern):
      yield os.path.join(dirpath, filename)


so_lib_paths = [
    i for i in os.listdir('.')
    if os.path.isdir(i) and fnmatch.fnmatch(i, '_solib_*')
]

print(os.listdir('.'))
matches = []
for path in so_lib_paths:
  matches.extend(
      ['../' + x for x in find_files('*', path) if '.py' not in x]
  )

if os.name == 'nt':
  EXTENSION_NAME = 'libamdcpu_plugin.pyd'
else:
  EXTENSION_NAME = 'libamdcpu_plugin.so'

headers = (
    list(find_files('*.h', 'tensorflow-plugins/c_api/c')) +
    list(find_files('*.h', 'tensorflow-plugins/c_api/src')))

curr_dir = os.path.dirname(__file__)
_tf_ver = os.environ.get('TF_VERSION', 'N/A')

long_description = ""
_desc_file = "DESCRIPTION_weekly.md" if _RELEASE_TYPE == 'weekly' else "DESCRIPTION.md"
with open(os.path.join(curr_dir, _desc_file), encoding="utf-8") as f:
  long_description = f.read()
long_description = long_description.replace('{{TF_VERSION}}', _tf_ver)

_source_tag = os.environ.get('ZENTF_SOURCE_TAG', '')
_tag_commit = os.environ.get('ZENTF_TAG_COMMIT', '')
_build_commit = os.environ.get('ZENTF_BUILD_COMMIT', '')

_GITHUB_REPO = "https://github.com/amd/ZenDNN-tensorflow-plugin"
_build_info_section = "\n## Build Information\n\n"
_build_info_section += "| Field | Value |\n|---|---|\n"
if _source_tag:
  _build_info_section += f"| Source Tag | `{_source_tag}` |\n"
  _build_info_section += f"| Tag Commit | `{_tag_commit[:12]}` |\n"
if _build_commit:
  _build_info_section += f"| Build Commit | `{_build_commit[:12]}` |\n"
_build_info_section += f"| TensorFlow Version | `{_tf_ver}` |\n"
_build_info_section += f"| Release Type | `{_RELEASE_TYPE}` |\n"
if _source_tag:
  _tag_url = f"{_GITHUB_REPO}/releases/tag/{_source_tag}"
  _build_info_section += f"\nBuilt from [{_source_tag}]({_tag_url})\n"
long_description += _build_info_section

_project_urls = {
    **({
        "Source Tag": f"{_GITHUB_REPO}/releases/tag/{_source_tag}"
    } if _source_tag else {}),
    "Source": _GITHUB_REPO,
}

setup(
    name=project_name,
    version=_VERSION.replace('-', ''),
    description="zenTF : A TensorFlow extension for AMD EPYC CPUs.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://developer.amd.com/zendnn',
    project_urls=_project_urls,
    author='AMD',
    author_email='zendnn.maintainers@amd.com',
    # Contained modules and scripts.
    packages=[_PLUGIN_LIB_PATH, _MY_PLUGIN_PATH],
    entry_points={
        'console_scripts': CONSOLE_SCRIPTS,
    },
    headers=headers,
    install_requires=REQUIRED_PACKAGES,
    tests_require=REQUIRED_PACKAGES + TEST_PACKAGES,
    package_data={
        _PLUGIN_LIB_PATH: [
            '*.so'
        ]+ matches,
        _MY_PLUGIN_PATH: [
            '*', '*/*'
        ]
    },
    zip_safe=False,
    distclass=BinaryDistribution,
    cmdclass={
        'install_headers': InstallHeaders,
        'install': InstallCommand,
    },
    # PyPI package information.
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    license='Apache 2.0',
    keywords='tensorflow tensor machine learning plugin ZenDNN AMD',
)
