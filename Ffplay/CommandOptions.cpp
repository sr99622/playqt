#include "CommandOptions.h"
#include "mainwindow.h"

CommandOptions::CommandOptions()
{
    options[0].name = "L";
    options[0].flags = OPT_EXIT;
    options[0].u.func_arg = show_license;
    options[0].help = "show_license";

    options[1].name = "h";
    options[1].flags = OPT_EXIT;
    options[1].u.func_arg = show_help;
    options[1].help = "show_help";
    options[1].argname = "topic";

    options[2].name = "?";
    options[2].flags = OPT_EXIT;
    options[2].u.func_arg = show_help;
    options[2].help = "show_help";
    options[2].argname = "topic";

    options[3].name = "help";
    options[3].flags = OPT_EXIT;
    options[3].u.func_arg = show_help;
    options[3].help = "show_help";
    options[3].argname = "topic";

    options[4].name = "-help";
    options[4].flags = OPT_EXIT;
    options[4].u.func_arg = show_help;
    options[4].help = "show_help";
    options[4].argname = "topic";

    options[5].name = "version";
    options[5].flags = OPT_EXIT;
    options[5].u.func_arg = show_version;
    options[5].help = "show version";

    options[6].name = "buildconf";
    options[6].flags = OPT_EXIT;
    options[6].u.func_arg = show_buildconf;
    options[6].help = "show build configuration";

    options[7].name = "formats";
    options[7].flags = OPT_EXIT;
    options[7].u.func_arg = show_formats;
    options[7].help = "show available formats";

    options[8].name = "muxers";
    options[8].flags = OPT_EXIT;
    options[8].u.func_arg = show_muxers;
    options[8].help = "show available muxers";

    options[9].name = "demuxers";
    options[9].flags = OPT_EXIT;
    options[9].u.func_arg = show_demuxers;
    options[9].help = "show available demuxers";

    options[10].name = "devices";
    options[10].flags = OPT_EXIT;
    options[10].u.func_arg = show_devices;
    options[10].help = "show available devices";

    options[11].name = "codecs";
    options[11].flags = OPT_EXIT;
    options[11].u.func_arg = show_codecs;
    options[11].help = "show available codecs";

    options[12].name = "decoders";
    options[12].flags = OPT_EXIT;
    options[12].u.func_arg = show_decoders;
    options[12].help = "show available decoders";

    options[13].name = "encoders";
    options[13].flags = OPT_EXIT;
    options[13].u.func_arg = show_encoders;
    options[13].help = "show available encoders";

    options[14].name = "bsfs";
    options[14].flags = OPT_EXIT;
    options[14].u.func_arg = show_bsfs;
    options[14].help = "show available bit stream filters";

    options[15].name = "protocols";
    options[15].flags = OPT_EXIT;
    options[15].u.func_arg = show_protocols;
    options[15].help = "show available protocols";

    options[16].name = "filters";
    options[16].flags = OPT_EXIT;
    options[16].u.func_arg = show_filters;
    options[16].help = "show available filters";

    options[17].name = "pix_fmts";
    options[17].flags = OPT_EXIT;
    options[17].u.func_arg = show_pix_fmts;
    options[17].help = "show available pixel formats";

    options[18].name = "layouts";
    options[18].flags = OPT_EXIT;
    options[18].u.func_arg = show_layouts;
    options[18].help = "show standard channel layouts";

    options[19].name = "sample_fmts";
    options[19].flags = OPT_EXIT;
    options[19].u.func_arg = show_sample_fmts;
    options[19].help = "show available audio sample formats";

    options[20].name = "colors";
    options[20].flags = OPT_EXIT;
    options[20].u.func_arg = show_colors;
    options[20].help = "show available color names";

    options[21].name = "loglevel";
    options[21].flags = HAS_ARG | OPT_NO_GUI;
    options[21].u.func_arg = opt_loglevel;
    options[21].help = "set logging level";
    options[21].argname = "loglevel";

    options[22].name = "v";
    options[22].flags = HAS_ARG | OPT_NO_GUI;
    options[22].u.func_arg = opt_loglevel;
    options[22].help = "set logging level";
    options[22].argname = "loglevel";

    options[23].name = "report";
    options[23].flags = 0 | OPT_NO_GUI;
    options[23].u.dst_ptr = (void*)opt_report;
    options[23].help = "generate a report";

    options[24].name = "max_alloc";
    options[24].flags = HAS_ARG | OPT_NO_GUI;
    options[24].u.func_arg = opt_max_alloc;
    options[24].help = "set maximum size of an allocated block";
    options[24].argname = "bytes";

    options[25].name = "cpuflags";
    options[25].flags = HAS_ARG | OPT_EXPERT | OPT_NO_GUI;
    options[25].u.func_arg = opt_cpuflags;
    options[25].help = "force specific cpu flags";
    options[25].argname = "flags";

    options[26].name = "hide_banner";
    options[26].flags = OPT_BOOL | OPT_EXPERT | OPT_NO_GUI;
    options[26].u.dst_ptr = {&hide_banner};
    options[26].help = "do not show program banner";
    options[26].argname = "hide_banner";

    options[27].name = "sources";
    options[27].flags = OPT_EXIT | HAS_ARG;
    options[27].u.func_arg = show_sources;
    options[27].help = "list sources of the input device";
    options[27].argname = "device";

    options[28].name = "sinks";
    options[28].flags = OPT_EXIT | HAS_ARG;
    options[28].u.func_arg = show_sinks;
    options[28].help = "list sinks of the output device";
    options[28].argname = "device";

    options[29].name = "x";
    options[29].flags = HAS_ARG | OPT_NO_GUI;
    options[29].u.func_arg = opt_width;
    options[29].help = "force displayed width";
    options[29].argname = "width";

    options[30].name = "y";
    options[30].flags = HAS_ARG | OPT_NO_GUI;
    options[30].u.func_arg = opt_height;
    options[30].help = "force displayed height";
    options[30].argname = "height";

    options[31].name = "s";
    options[31].flags = HAS_ARG | OPT_VIDEO | OPT_NO_GUI;
    options[31].u.func_arg = opt_frame_size;
    options[31].help = "set frame size (WxH or abbreviation)";
    options[31].argname = "size";

    options[32].name = "fs";
    options[32].flags = OPT_BOOL | OPT_NO_GUI;
    options[32].u.dst_ptr = &is_full_screen;
    options[32].help = "force full screen";

    options[33].name = "an";
    options[33].flags = OPT_BOOL;
    options[33].u.dst_ptr = &audio_disable;
    options[33].help = "disable audio";

    options[34].name = "vn";
    options[34].flags = OPT_BOOL | OPT_NO_GUI;
    options[34].u.dst_ptr = &video_disable;
    options[34].help = "disable video";

    options[35].name = "sn";
    options[35].flags = OPT_BOOL | OPT_NO_GUI;
    options[35].u.dst_ptr = &subtitle_disable;
    options[35].help = "disable subtitling";

    options[36].name = "ast";
    options[36].flags = OPT_STRING | HAS_ARG | OPT_EXPERT;
    options[36].u.dst_ptr =  &wanted_stream_spec[AVMEDIA_TYPE_AUDIO];
    options[36].help = "select desired audio stream";
    options[36].argname = "stream_specifier";

    options[37].name = "vst";
    options[37].flags = OPT_STRING | HAS_ARG | OPT_EXPERT;
    options[37].u.dst_ptr = &wanted_stream_spec[AVMEDIA_TYPE_VIDEO];
    options[37].help = "select desired video stream";
    options[37].argname = "stream_specifier";

    options[38].name = "sst";
    options[38].flags = OPT_STRING | HAS_ARG | OPT_EXPERT;
    options[38].u.dst_ptr = &wanted_stream_spec[AVMEDIA_TYPE_SUBTITLE];
    options[38].help = "select desired subtitle stream";
    options[38].argname = "stream_specifier";

    options[39].name = "ss";
    options[39].flags = HAS_ARG;
    options[39].u.func_arg = opt_seek;
    options[39].help = "seek to a position in seconds";
    options[39].argname = "pos";

    options[40].name = "t";
    options[40].flags = HAS_ARG;
    options[40].u.func_arg = opt_duration;
    options[40].help = "play duration in seconds";
    options[40].argname = "duration";

    options[41].name = "bytes";
    options[41].flags = OPT_INT | HAS_ARG;
    options[41].u.dst_ptr = &seek_by_bytes;
    options[41].help = "seek by bytes";
    options[41].argname = "val";

    options[42].name = "seek_interval";
    options[42].flags = OPT_FLOAT | HAS_ARG;
    options[42].u.dst_ptr = &seek_interval;
    options[42].help = "set seek interval in seconds";
    options[42].argname = "seconds";

    options[43].name = "nodisp";
    options[43].flags = OPT_BOOL | OPT_NO_GUI;
    options[43].u.dst_ptr = &display_disable;
    options[43].help = "disable graphical display";

    options[44].name = "noborder";
    options[44].flags = OPT_BOOL | OPT_NO_GUI;
    options[44].u.dst_ptr = &borderless;
    options[44].help = "borderless window";

    options[45].name = "volume";
    options[45].flags = OPT_INT | HAS_ARG | OPT_NO_GUI;
    options[45].u.dst_ptr = &startup_volume;
    options[45].help = "set startup volume";
    options[45].argname = "volume";

    options[46].name = "f";
    options[46].flags = HAS_ARG;
    options[46].u.func_arg = opt_format;
    options[46].help = "force format";
    options[46].argname = "fmt";

    options[47].name = "pix_fmt";
    options[47].flags = HAS_ARG | OPT_EXPERT | OPT_VIDEO | OPT_NO_GUI;
    options[47].u.func_arg = opt_frame_pix_fmt;
    options[47].help = "set pixel format";
    options[47].argname = "format";

    options[48].name = "stats";
    options[48].flags = OPT_BOOL | OPT_EXPERT | OPT_NO_GUI;
    options[48].u.dst_ptr = &show_status;
    options[48].help = "show status";
    options[48].argname = "";

    options[49].name = "fast";
    options[49].flags = OPT_BOOL | OPT_EXPERT;
    options[49].u.dst_ptr = &fast;
    options[49].help = "non spec compliant optimizations";
    options[49].argname = "";

    options[50].name = "genpts";
    options[50].flags = OPT_BOOL | OPT_EXPERT;
    options[50].u.dst_ptr = &genpts;
    options[50].help = "generate pts";
    options[50].argname = "";

    options[51].name = "drp";
    options[51].flags = OPT_INT | HAS_ARG | OPT_EXPERT;
    options[51].u.dst_ptr = &decoder_reorder_pts;
    options[51].help = "let decoder reorder pts";
    options[51].argname = "";

    options[52].name = "lowres";
    options[52].flags = OPT_INT | HAS_ARG | OPT_EXPERT | OPT_NO_GUI;
    options[52].u.dst_ptr = &lowres;
    options[52].help = "low resolution";
    options[52].argname = "";

    options[53].name = "sync";
    options[53].flags = HAS_ARG | OPT_EXPERT;
    options[53].u.func_arg = opt_sync;
    options[53].help = "set audio-video sync";
    options[53].argname = "type";

    options[54].name = "autoexit";
    options[54].flags = OPT_BOOL | OPT_EXPERT | OPT_NO_GUI;
    options[54].u.dst_ptr = &autoexit;
    options[54].help = "exit at the end";
    options[54].argname = "";

    options[55].name = "exitonkeydown";
    options[55].flags = OPT_BOOL | OPT_EXPERT | OPT_NO_GUI;
    options[55].u.dst_ptr = &exit_on_keydown;
    options[55].help = "exit on key down";
    options[55].argname = "";

    options[56].name = "exitonmousedown";
    options[56].flags = OPT_BOOL | OPT_EXPERT | OPT_NO_GUI;
    options[56].u.dst_ptr = &exit_on_mousedown;
    options[56].help = "exit on mouse down";
    options[56].argname = "";

    options[57].name = "loop";
    options[57].flags = OPT_INT | HAS_ARG | OPT_EXPERT | OPT_NO_GUI;
    options[57].u.dst_ptr = &loop;
    options[57].help = "set number of times the playback shall be looped";
    options[57].argname = "loop count";

    options[58].name = "framedrop";
    options[58].flags = OPT_BOOL | OPT_EXPERT;
    options[58].u.dst_ptr = &framedrop;
    options[58].help = "drop frames when cpu is too slow";
    options[58].argname = "";

    options[59].name = "infbuf";
    options[59].flags = OPT_BOOL | OPT_EXPERT;
    options[59].u.dst_ptr = &infinite_buffer;
    options[59].help = "infinite input buffer";
    options[59].argname = "";

    options[60].name = "window_title";
    options[60].flags = OPT_STRING | HAS_ARG | OPT_NO_GUI;
    options[60].u.dst_ptr = &window_title;
    options[60].help = "set window title";
    options[60].argname = "window title";

    options[61].name = "left";
    options[61].flags = OPT_INT | HAS_ARG | OPT_EXPERT | OPT_NO_GUI;
    options[61].u.dst_ptr = &screen_left;
    options[61].help = "set the x position for the left of the window";
    options[61].argname = "x pos";

    options[62].name = "top";
    options[62].flags = OPT_INT | HAS_ARG | OPT_EXPERT | OPT_NO_GUI;
    options[62].u.dst_ptr = &screen_top;
    options[62].help = "set the y position for the top of the window";
    options[62].argname = "y pos";

#if CONFIG_AVFILTER
    options[63].name = "vf";
    options[63].flags = OPT_EXPERT | HAS_ARG;
    options[63].u.func_arg = opt_add_vfilter;
    options[63].help = "set video filters";
    options[63].argname = "filter_graph";
#endif

    options[64].name = "af";
    options[64].flags = OPT_STRING | HAS_ARG;
    options[64].u.dst_ptr = &afilters;
    options[64].help = "set audio filters";
    options[64].argname = "filter_graph";

    //{ "rdftspeed", OPT_INT | HAS_ARG | OPT_AUDIO | OPT_EXPERT, { &rdftspeed }, "rdft speed", "msecs" },

    options[65].name = "showmode";
    options[65].flags = HAS_ARG | OPT_NO_GUI;
    options[65].u.func_arg = opt_show_mode;
    options[65].help = "select show mode (0 = video, 1 = waves, 2 = RDFT)";
    options[65].argname = "mode";

    options[66].name = "default";
    options[66].flags = HAS_ARG | OPT_AUDIO | OPT_VIDEO | OPT_EXPERT | OPT_NO_GUI;
    options[66].u.func_arg = opt_default;
    options[66].help = "generic catch all option";
    options[66].argname = "";

    options[67].name = "i";
    options[67].flags = OPT_BOOL | OPT_NO_GUI;
    options[67].u.dst_ptr = &dummy;
    options[67].help = "read specified file";
    options[67].argname = "input_file";

    options[68].name = "codec";
    options[68].flags = HAS_ARG | OPT_NO_GUI;
    options[68].u.func_arg = opt_codec;
    options[68].help = "force decoder";
    options[68].argname = "decoder_name";

    options[69].name = "acodec";
    options[69].flags = HAS_ARG | OPT_STRING | OPT_EXPERT;
    options[69].u.dst_ptr = &audio_codec_name;
    options[69].help = "force audio decoder";
    options[69].argname = "decoder_name";

    options[70].name = "scodec";
    options[70].flags = HAS_ARG | OPT_STRING | OPT_EXPERT;
    options[70].u.dst_ptr = &subtitle_codec_name;
    options[70].help = "force subtitle decoder";
    options[70].argname = "decoder_name";

    options[71].name = "vcodec";
    options[71].flags = HAS_ARG | OPT_STRING | OPT_EXPERT;
    options[71].u.dst_ptr = &video_codec_name;
    options[71].help = "force video decoder";
    options[71].argname = "decoder_name";

    options[72].name = "autorotate";
    options[72].flags = OPT_BOOL | OPT_NO_GUI;
    options[72].u.dst_ptr = &autorotate;
    options[72].help = "automatically rotate video";
    options[72].argname = "";

    options[73].name = "find_stream_info";
    options[73].flags = OPT_BOOL | OPT_INPUT | OPT_EXPERT | OPT_NO_GUI;
    options[73].u.dst_ptr = &find_stream_info;
    options[73].help = "find stream information";

    options[74].name = "filter_threads";
    options[74].flags = HAS_ARG | OPT_INT | OPT_EXPERT | OPT_NO_GUI;
    options[74].u.dst_ptr = &filter_nbthreads;
    options[74].help = "number of filter threads per graph";

    //options[75] = NULL;
}

int CommandOptions::findOptionIndexByHelp(QString help)
{
    int result = -1;
    for (int i = 0; i < NUM_OPTIONS; i++) {
        if (help == options[i].help) {
            result = i;
            break;
        }
    }
    return result;
}

int CommandOptions::findOptionIndexByName(QString name)
{
    int result = -1;
    for (int i = 0; i < NUM_OPTIONS; i++) {
        if (name == options[i].name) {
            result = i;
            break;
        }
    }
    return result;
}

void CommandOptions::show_log_level(int log_level)
{
    switch (log_level) {
    case AV_LOG_QUIET:
        cout << "AV_LOG_QUIET" <<endl;
        break;
    case AV_LOG_PANIC:
        cout << "AV_LOG_PANIC" <<endl;
        break;
    case AV_LOG_FATAL:
        cout << "AV_LOG_FATAL" <<endl;
        break;
    case AV_LOG_ERROR:
        cout << "AV_LOG_ERROR" <<endl;
        break;
    case AV_LOG_WARNING:
        cout << "AV_LOG_WARNING" <<endl;
        break;
    case AV_LOG_VERBOSE:
        cout << "AV_LOG_VERBOSE" <<endl;
        break;
    case AV_LOG_DEBUG:
        cout << "AV_LOG_DEBUG" <<endl;
        break;
    case AV_LOG_TRACE:
        cout << "AV_LOG_TRACE" <<endl;
        break;
    }
}

void CommandOptions::showHelpCallback(void *ptr, int level, const char *fmt, va_list vl)
{
    if (!av_log_on)
        return;

    if (level == AV_LOG_INFO) {
        char buffer[256];
        vsprintf_s(buffer, 256, fmt, vl);

        QString str(buffer);
        emit showHelp(str);
    }
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
    forced_format = arg;

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
    clock_sync = av_strdup(arg);

    if (!strcmp(arg, "audio"))
        av_sync_type = AV_SYNC_AUDIO_MASTER;
    else if (!strcmp(arg, "video"))
        av_sync_type = AV_SYNC_VIDEO_MASTER;
    else if (!strcmp(arg, "ext"))
        av_sync_type = AV_SYNC_EXTERNAL_CLOCK;
    else {
        QString msg = "Invalid option for sync.  Use audio, video or ext";
        QMessageBox::warning(MW->parameterDialog, "Invalid setting", msg);
        av_sync_type = AV_SYNC_AUDIO_MASTER;
        clock_sync = nullptr;
        MW->parameter()->parameter->setText("");
        //((ParameterPanel*)MW->parameterDialog->panel)->parameter->setText("");
        //av_log(NULL, AV_LOG_ERROR, "Unknown value for %s: %s\n", opt, arg);
        //exit(1);
        return -1;
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
        char buf[256];
        sprintf(buf, "No media specifier was specified in '%s' in option '%s'\n", arg, opt);
        MW->msg(buf);
        return AVERROR(EINVAL);
    }
    spec++;
    switch (spec[0]) {
    case 'a':    audio_codec_name = arg; break;
    case 's': subtitle_codec_name = arg; break;
    case 'v':    video_codec_name = arg; break;
    default:
        char buf[256];
        sprintf(buf, "Invalid media specifier '%s' in option '%s'\n", spec, opt);
        return AVERROR(EINVAL);
    }
    return 0;
}

#if CONFIG_AVFILTER
int CommandOptions::opt_add_vfilter(void* optctx, const char* opt, const char* arg)
{
    //const char *filter_string = av_strdup(arg);

    if (strlen(arg) == 0) {
        for (int i = nb_vfilters - 1; i > -1; i--) {
            cout << vfilters_list[i] << endl;
            av_free(&vfilters_list[i]);
        }

        if (vfilters_list)
            av_free(&vfilters_list);

        vfilters_list = NULL;
        nb_vfilters = 0;
    }
    else {
        if (nb_vfilters == 0)
            GROW_ARRAY(vfilters_list, nb_vfilters);

        vfilters_list[nb_vfilters - 1] = av_strdup(arg);
    }

    return 0;
}
#endif

void CommandOptions::show_usage(void)
{
    av_log(NULL, AV_LOG_INFO, "Simple media player\n");
    av_log(NULL, AV_LOG_INFO, "usage: %s [options] input_file\n", program_name);
    av_log(NULL, AV_LOG_INFO, "\n");
}

