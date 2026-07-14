"""Point the Vulkan loader at the bundled MoltenVK ICD on macOS.

The Vulkan loader discovers drivers via ICD manifests in system paths
(/usr/local/share/vulkan/icd.d, ...) that only exist if the user installed
MoltenVK themselves. We ship MoltenVK inside the bundle (see smartloop.spec),
so aim the loader at it explicitly — unless the user already configured their
own driver via VK_DRIVER_FILES / VK_ICD_FILENAMES.

On Linux/Windows the GPU driver registers its own ICD; nothing to do.
"""
import os
import sys

if sys.platform == 'darwin' and not (
    os.environ.get('VK_DRIVER_FILES') or os.environ.get('VK_ICD_FILENAMES')
):
    _icd = os.path.join(
        getattr(sys, '_MEIPASS', ''), 'llama_cpp', 'lib', 'MoltenVK_icd.json'
    )
    if os.path.isfile(_icd):
        os.environ['VK_DRIVER_FILES'] = _icd
        # Older loaders only honor the legacy name.
        os.environ['VK_ICD_FILENAMES'] = _icd
