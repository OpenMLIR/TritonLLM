from triton.runtime import driver
import triton.language as tl

__all__ = ["current_target"]


def current_target():
    try:
        active_driver = driver.active
    except RuntimeError:
        # If there is no active driver, return None
        return None
    return active_driver.get_current_target()


current_target.__triton_builtin__ = True


@tl.constexpr_function
def is_cuda():
    target = current_target()
    return target is not None and target.backend == "cuda"


@tl.constexpr_function
def cuda_capability_geq(major, minor=0):
    """
    Determines whether we have compute capability >= (major, minor) and
    returns this as a constexpr boolean. This can be used for guarding
    inline asm implementations that require a certain compute capability.
    """
    target = current_target()
    if target is None or target.backend != "cuda":
        return False
    assert isinstance(target.arch, int)
    return target.arch >= major * 10 + minor


@tl.constexpr_function
def is_hip():
    target = current_target()
    return target is not None and target.backend == "hip"


@tl.constexpr_function
def is_hip_cdna3():
    target = current_target()
    return target is not None and target.arch == "gfx942"


@tl.constexpr_function
def is_hip_cdna4():
    target = current_target()
    return target is not None and target.arch == "gfx950"
