/*******************************************************************************
* avexception.cpp
*
* Copyright (c) 2020 Stephen Rhodes
*
* This program is free software; you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation; either version 2 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along
* with this program; if not, write to the Free Software Foundation, Inc.,
* 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*
*******************************************************************************/

#include "avexception.h"
#include <QTextStream>

AVException::AVException(int id, int tag)
{
    if (id < 0) {
        error_id = id;
        cmd_tag = tag;
        if (error_id == NULL_ERROR)
            strcpy(error_text, "Function call returned NULL");
        else
            av_strerror(error_id, error_text, 1024);
        throw this;
    }
}

AVExceptionHandler::AVExceptionHandler()
{

}

void AVExceptionHandler::ck(int ret, int tag = 0)
{
    AVException e(ret, tag);
}

QString AVExceptionHandler::tag(int cmd_tag)
{
    switch (cmd_tag) {
    case AO2:
        return "avcodec_open2";
    case AOI:
        return "avformat_open_input";
    case ACI:
        return "avformat_close_input";
    case AFSI:
        return "avformat_find_stream_info";
    case AFBS:
        return "av_find_best_stream";
    case APTC:
        return "avcodec_parameters_to_context";
    case APFC:
        return "avcodec_parameters_from_context";
    case AWH:
        return "av_write_header";
    case AWT:
        return "av_write_trailer";
    case AO:
        return "avio_open";
    case AC:
        return "avio_close";
    case ACP:
        return "avio_closep";
    case AAOC2:
        return "avformat_alloc_output_context2";
    case AFMW:
        return "av_frame_make_writable";
    case AFGB:
        return "av_frame_get_buffer";
    case AHCC:
        return "av_hwdevice_ctx_create";
    case AWF:
        return "av_write_frame";
    case ASP:
        return "avcodec_send_packet";
    case ASF:
        return "av_seek_frame";
    case AEV2:
        return "avcodec_encode_video2";
    case ARF:
        return "av_read_frame";
    case ADV2:
        return "av_decode_video2";
    case ARP:
        return "avcodec_recieve_packet";
    case AIWF:
        return "av_interleaved_write_frame";
    case AFE:
        return "avcodec_find_encoder";
    case AAC3:
        return "avcodec_alloc_context3";
    case AFA:
        return "av_alloc_frame";
    case AAC:
        return "avformat_alloc_context";
    case AFC:
        return "av_frame_copy";
    case ABR:
        return "av_buffer_ref";
    case AHFTBN:
        return "av_hwdevice_find_type_by_name";
    case AGHC:
        return "avcodec_get_hw_config";
    case ANS:
        return "avformat_new_stream";
    case SGC:
        return "sws_getContext";
    case PTF:
        return "Picture::toFrame";
    case GACC:
        return "MainWindow::getActiveCodecContext";
    case AFIF:
        return "av_find_input_format";
    default:
        return "";
    }
}

QString AVExceptionHandler::contextToString(AVCodecContext *arg)
{
    QString str = "\n";
    switch (arg->codec_id) {
    case AV_CODEC_ID_H264:
        str.append("codec: AV_CODEC_ID_H264\n");
        break;
    case AV_CODEC_ID_HEVC:
        str.append("codec: AV_CODEC_ID_HEVC\n");
        break;
    case AV_CODEC_ID_MJPEG:
        str.append("codec: AV_CODEC_ID_MJPEG\n");
        break;
    default:
        str.append("codec: UNKNOWN\n");
    }
    switch (arg->pix_fmt) {
    case AV_PIX_FMT_YUV420P:
        str.append("pix_fmt: AV_PIX_FMT_YUV420P");
        break;
    case AV_PIX_FMT_YUVJ420P:
        str.append("pix_fmt: AV_PIX_FMT_YUVJ420P");
        break;
    case AV_PIX_FMT_RGB8:
        str.append("pix_fmt: AVPIX_FMT_RGB8");
        break;
    default:
        str.append("pix_fmt: UNKNOWN");
    }

    QTextStream(&str) << "\n"
                      << "width: " << arg->width << "\n"
                      << "height: " << arg->height << "\n"
                      << "time_base.num: " << arg->time_base.num << "\n"
                      << "time_base.den: " << arg->time_base.den << "\n"
                      << "framerate.num: " << arg->framerate.num << "\n"
                      << "framerate.den: " << arg->framerate.den << "\n"
                      << "bit_rate: " << arg->bit_rate << "\n"
                      << "gop_size: " << arg->gop_size << "\n"
                      << "max_b_frames: " << arg->max_b_frames << "\n"
                      << "extradata_size: " << arg->extradata_size << "\n";
    return str;
}
