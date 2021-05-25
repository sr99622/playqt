#include "CommandOptions.h"

CommandOptions::CommandOptions()
{
}

int CommandOptions::opt_frame_size(void* optctx, const char* opt, const char* arg)
{
    av_log(NULL, AV_LOG_WARNING, "Option -s is deprecated, use -video_size.\n");
    return opt_default(NULL, "video_size", arg);
}

int CommandOptions::opt_width(void* optctx, const char* opt, const char* arg)
{
    screen_width = parse_number_or_die(opt, arg, OPT_INT64, 1, INT_MAX);
    return 0;
}

int CommandOptions::opt_height(void* optctx, const char* opt, const char* arg)
{
    screen_height = parse_number_or_die(opt, arg, OPT_INT64, 1, INT_MAX);
    return 0;
}

int CommandOptions::opt_format(void* optctx, const char* opt, const char* arg)
{
    file_iformat = av_find_input_format(arg);
    if (!file_iformat) {
        av_log(NULL, AV_LOG_FATAL, "Unknown input format: %s\n", arg);
        return AVERROR(EINVAL);
    }
    return 0;
}

int CommandOptions::opt_frame_pix_fmt(void* optctx, const char* opt, const char* arg)
{
    av_log(NULL, AV_LOG_WARNING, "Option -pix_fmt is deprecated, use -pixel_format.\n");
    return opt_default(NULL, "pixel_format", arg);
}

int CommandOptions::opt_sync(void* optctx, const char* opt, const char* arg)
{
    if (!strcmp(arg, "audio"))
        av_sync_type = AV_SYNC_AUDIO_MASTER;
    else if (!strcmp(arg, "video"))
        av_sync_type = AV_SYNC_VIDEO_MASTER;
    else if (!strcmp(arg, "ext"))
        av_sync_type = AV_SYNC_EXTERNAL_CLOCK;
    else {
        av_log(NULL, AV_LOG_ERROR, "Unknown value for %s: %s\n", opt, arg);
        exit(1);
    }
    return 0;
}

int CommandOptions::opt_seek(void* optctx, const char* opt, const char* arg)
{
    start_time = parse_time_or_die(opt, arg, 1);
    return 0;
}

int CommandOptions::opt_duration(void* optctx, const char* opt, const char* arg)
{
    duration = parse_time_or_die(opt, arg, 1);
    return 0;
}

int CommandOptions::opt_show_mode(void* optctx, const char* opt, const char* arg)
{
    show_mode = (ShowMode)(!strcmp(arg, "video") ? SHOW_MODE_VIDEO :
        //!strcmp(arg, "waves") ? SHOW_MODE_WAVES :
        !strcmp(arg, "rdft") ? SHOW_MODE_RDFT :
        parse_number_or_die(opt, arg, OPT_INT, 0, SHOW_MODE_NB - 1));
    return 0;
}

void CommandOptions::opt_input_file(void* optctx, const char* filename)
{
    if (input_filename) {
        av_log(NULL, AV_LOG_FATAL,
            "Argument '%s' provided as input filename, but '%s' was already specified.\n",
            filename, input_filename);
        exit(1);
    }
    if (!strcmp(filename, "-"))
        filename = "pipe:";
    input_filename = filename;
}

int CommandOptions::opt_codec(void* optctx, const char* opt, const char* arg)
{
    const char* spec = strchr(opt, ':');
    if (!spec) {
        av_log(NULL, AV_LOG_ERROR,
            "No media specifier was specified in '%s' in option '%s'\n",
            arg, opt);
        return AVERROR(EINVAL);
    }
    spec++;
    switch (spec[0]) {
    case 'a':    audio_codec_name = arg; break;
    case 's': subtitle_codec_name = arg; break;
    case 'v':    video_codec_name = arg; break;
    default:
        av_log(NULL, AV_LOG_ERROR,
            "Invalid media specifier '%s' in option '%s'\n", spec, opt);
        return AVERROR(EINVAL);
    }
    return 0;
}

#if CONFIG_AVFILTER
int CommandOptions::opt_add_vfilter(void* optctx, const char* opt, const char* arg)
{
    GROW_ARRAY(vfilters_list, nb_vfilters);
    vfilters_list[nb_vfilters - 1] = arg;
    return 0;
}
#endif

void CommandOptions::show_usage(void)
{
    av_log(NULL, AV_LOG_INFO, "Simple media player\n");
    av_log(NULL, AV_LOG_INFO, "usage: %s [options] input_file\n", program_name);
    av_log(NULL, AV_LOG_INFO, "\n");
}

