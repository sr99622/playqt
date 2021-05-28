#include "npperror.h"

NppError::NppError()
{

}

QString NppError::getError(const NppStatus& status)
{
    QString result;
    switch (status) {
    case NPP_NOT_SUPPORTED_MODE_ERROR:
        result = "NPP_NOT_SUPPORTED_MODE_ERROR";
        break;
    case NPP_INVALID_HOST_POINTER_ERROR:
        result = "NPP_INVALID_HOST_POINTER_ERROR";
        break;
    case NPP_INVALID_DEVICE_POINTER_ERROR:
        result = "NPP_INVALID_DEVICE_POINTER_ERROR";
        break;
    case NPP_LUT_PALETTE_BITSIZE_ERROR:
        result = "NPP_LUT_PALETTE_BITSIZE_ERROR";
        break;
    case NPP_ZC_MODE_NOT_SUPPORTED_ERROR:
        result = "NPP_ZC_MODE_NOT_SUPPORTED_ERROR";
        break;
    case NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY:
        result = "NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY";
        break;
    case NPP_TEXTURE_BIND_ERROR:
        result = "NPP_TEXTURE_BIND_ERROR";
        break;
    case NPP_WRONG_INTERSECTION_ROI_ERROR:
        result = "NPP_WRONG_INTERSECTION_ROI_ERROR";
        break;
    case NPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR:
        result = "NPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR";
        break;
    case NPP_MEMFREE_ERROR:
        result = "NPP_MEMFREE_ERROR";
        break;
    case NPP_MEMSET_ERROR:
        result = "NPP_MEMSET_ERROR";
        break;
    case NPP_MEMCPY_ERROR:
        result = "NPP_MEMCPY_ERROR";
        break;
    case NPP_ALIGNMENT_ERROR:
        result = "NPP_ALIGNMENT_ERROR";
        break;
    case NPP_CUDA_KERNEL_EXECUTION_ERROR:
        result = "NPP_CUDA_KERNEL_EXECUTION_ERROR";
        break;
    case NPP_ROUND_MODE_NOT_SUPPORTED_ERROR:
        result = "NPP_ROUND_MODE_NOT_SUPPORTED_ERROR";
        break;
    case NPP_QUALITY_INDEX_ERROR:
        result = "NPP_QUALITY_INDEX_ERROR";
        break;
    case NPP_RESIZE_NO_OPERATION_ERROR:
        result = "NPP_RESIZE_NO_OPERATION_ERROR";
        break;
    case NPP_OVERFLOW_ERROR:
        result = "NPP_OVERFLOW_ERROR";
        break;
    case NPP_NOT_EVEN_STEP_ERROR:
        result = "NPP_NOT_EVEN_STEP_ERROR";
        break;
    case NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR:
        result = "NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR";
        break;
    case NPP_LUT_NUMBER_OF_LEVELS_ERROR:
        result = "NPP_LUT_NUMBER_OF_LEVELS_ERROR";
        break;
    case NPP_CORRUPTED_DATA_ERROR:
        result = "NPP_CORRUPTED_DATA_ERROR";
        break;
    case NPP_CHANNEL_ORDER_ERROR:
        result = "NPP_CHANNEL_ORDER_ERROR";
        break;
    case NPP_ZERO_MASK_VALUE_ERROR:
        result = "NPP_ZERO_MASK_VALUE_ERROR";
        break;
    case NPP_QUADRANGLE_ERROR:
        result = "NPP_QUADRANGLE_ERROR";
        break;
    case NPP_RECTANGLE_ERROR:
        result = "NPP_RECTANGLE_ERROR";
        break;
    case NPP_COEFFICIENT_ERROR:
        result = "NPP_COEFFICIENT_ERROR";
        break;
    case NPP_NUMBER_OF_CHANNELS_ERROR:
        result = "NPP_NUMBER_OF_CHANNELS_ERROR";
        break;
    case NPP_COI_ERROR:
        result = "NPP_COI_ERROR";
        break;
    case NPP_DIVISOR_ERROR:
        result = "NPP_DIVISOR_ERROR";
        break;
    case NPP_CHANNEL_ERROR:
        result = "NPP_CHANNEL_ERROR";
        break;
    case NPP_STRIDE_ERROR:
        result = "NPP_STRIDE_ERROR";
        break;
    case NPP_ANCHOR_ERROR:
        result = "NPP_ANCHOR_ERROR";
        break;
    case NPP_MASK_SIZE_ERROR:
        result = "NPP_MASK_SIZE_ERROR";
        break;
    case NPP_RESIZE_FACTOR_ERROR:
        result = "NPP_RESIZE_FACTOR_ERROR";
        break;
    case NPP_INTERPOLATION_ERROR:
        result = "NPP_INTERPOLATION_ERROR";
        break;
    case NPP_MIRROR_FLIP_ERROR:
        result = "NPP_MIRROR_FLIP_ERROR";
        break;
    case NPP_MOMENT_00_ZERO_ERROR:
        result = "NPP_MOMENT_00_ZERO_ERROR";
        break;
    case NPP_THRESHOLD_NEGATIVE_LEVEL_ERROR:
        result = "NPP_THRESHOLD_NEGATIVE_LEVEL_ERROR";
        break;
    case NPP_THRESHOLD_ERROR:
        result = "NPP_THRESHOLD_ERROR";
        break;
    case NPP_CONTEXT_MATCH_ERROR:
        result = "NPP_CONTEXT_MATCH_ERROR";
        break;
    case NPP_FFT_FLAG_ERROR:
        result = "NPP_FFT_FLAG_ERROR";
        break;
    case NPP_FFT_ORDER_ERROR:
        result = "NPP_FFT_ORDER_ERROR";
        break;
    case NPP_STEP_ERROR:
        result = "NPP_STEP_ERROR";
        break;
    case NPP_SCALE_RANGE_ERROR:
        result = "NPP_SCALE_RANGE_ERROR";
        break;
    case NPP_DATA_TYPE_ERROR:
        result = "NPP_DATA_TYPE_ERROR";
        break;
    case NPP_OUT_OFF_RANGE_ERROR:
        result = "NPP_OUT_OFF_RANGE_ERROR";
        break;
    case NPP_DIVIDE_BY_ZERO_ERROR:
        result = "NPP_DIVIDE_BY_ZERO_ERROR";
        break;
    case NPP_MEMORY_ALLOCATION_ERR:
        result = "NPP_MEMORY_ALLOCATION_ERR";
        break;
    case NPP_NULL_POINTER_ERROR:
        result = "NPP_NULL_POINTER_ERROR";
        break;
    case NPP_RANGE_ERROR:
        result = "NPP_RANGE_ERROR";
        break;
    case NPP_SIZE_ERROR:
        result = "NPP_SIZE_ERROR";
        break;
    case NPP_BAD_ARGUMENT_ERROR:
        result = "NPP_BAD_ARGUMENT_ERROR";
        break;
    case NPP_NO_MEMORY_ERROR:
        result = "NPP_NO_MEMORY_ERROR";
        break;
    case NPP_NOT_IMPLEMENTED_ERROR:
        result = "NPP_NOT_IMPLEMENTED_ERROR";
        break;
    case NPP_ERROR:
        result = "NPP_ERROR";
        break;
    case NPP_ERROR_RESERVED:
        result = "NPP_ERROR_RESERVED";
        break;
    case NPP_NO_ERROR:
        result = "NPP_NO_ERROR";
        break;
    case NPP_NO_OPERATION_WARNING:
        result = "NPP_NO_OPERATION_WARNING";
        break;
    case NPP_DIVIDE_BY_ZERO_WARNING:
        result = "NPP_DIVIDE_BY_ZERO_WARNING";
        break;
    case NPP_AFFINE_QUAD_INCORRECT_WARNING:
        result = "NPP_AFFINE_QUAD_INCORRECT_WARNING";
        break;
    case NPP_WRONG_INTERSECTION_ROI_WARNING:
        result = "NPP_WRONG_INTERSECTION_ROI_WARNING";
        break;
    case NPP_WRONG_INTERSECTION_QUAD_WARNING:
        result = "NPP_WRONG_INTERSECTION_QUAD_WARNING";
        break;
    case NPP_DOUBLE_SIZE_WARNING:
        result = "NPP_DOUBLE_SIZE_WARNING";
        break;
    case NPP_MISALIGNED_DST_ROI_WARNING:
        result = "NPP_MISALIGNED_DST_ROI_WARNING";
        break;
    }

    return result;
}
