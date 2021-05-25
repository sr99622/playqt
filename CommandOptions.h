#pragma once

extern"C" {
#include "cmdutils.h"
#include "libavutil/opt.h"
}

#include <SDL.h>

//#include "VideoState.h"

//#define CO ((CommandOptions*)command_options)

enum ShowMode {
    SHOW_MODE_NONE = -1, 
    SHOW_MODE_VIDEO = 0, 
    SHOW_MODE_RDFT, 
    SHOW_MODE_NB
};

enum SyncMode {
    AV_SYNC_AUDIO_MASTER, // default choice 
    AV_SYNC_VIDEO_MASTER,
    AV_SYNC_EXTERNAL_CLOCK, // synchronize to an external clock 
};


#define SAMPLE_ARRAY_SIZE (8 * 65536)
#define USE_ONEPASS_SUBTITLE_RENDER 1
#define SDL_AUDIO_MIN_BUFFER_SIZE 512
#define SDL_AUDIO_MAX_CALLBACKS_PER_SEC 30
#define AUDIO_DIFF_AVG_NB   20
#define SAMPLE_CORRECTION_PERCENT_MAX 10
#define EXTERNAL_CLOCK_MIN_FRAMES 2
#define EXTERNAL_CLOCK_MAX_FRAMES 10
#define EXTERNAL_CLOCK_SPEED_MIN  0.900
#define EXTERNAL_CLOCK_SPEED_MAX  1.010
#define EXTERNAL_CLOCK_SPEED_STEP 0.001

class CommandOptions
{
public:
    CommandOptions();

    static void show_usage(void);

    static int opt_frame_size(void* optctx, const char* opt, const char* arg);
    static int opt_width(void* optctx, const char* opt, const char* arg);
    static int opt_height(void* optctx, const char* opt, const char* arg);
    static int opt_format(void* optctx, const char* opt, const char* arg);
    static int opt_frame_pix_fmt(void* optctx, const char* opt, const char* arg);
    static int opt_sync(void* optctx, const char* opt, const char* arg);
    static int opt_seek(void* optctx, const char* opt, const char* arg);
    static int opt_duration(void* optctx, const char* opt, const char* arg);
    static int opt_show_mode(void* optctx, const char* opt, const char* arg);
    static void opt_input_file(void* optctx, const char* filename);
    static int opt_codec(void* optctx, const char* opt, const char* arg);

#if CONFIG_AVFILTER
    static int opt_add_vfilter(void* optctx, const char* opt, const char* arg);
#endif

    inline static int dummy;

    inline static AVInputFormat* file_iformat;
    inline static const char* input_filename;
    inline static const char* window_title;
    inline static int default_width = 640;
    inline static int default_height = 480;
    inline static int screen_width = 0;
    inline static int screen_height = 0;
    inline static int screen_left = SDL_WINDOWPOS_CENTERED;
    inline static int screen_top = SDL_WINDOWPOS_CENTERED;
    inline static int audio_disable;
    inline static int video_disable;
    inline static int subtitle_disable;
    inline static const char* wanted_stream_spec[AVMEDIA_TYPE_NB] = { 0 };
    inline static int seek_by_bytes = -1;
    inline static float seek_interval = 10;
    inline static int display_disable;
    inline static int borderless;
    inline static int startup_volume = 100;
    inline static int show_status = 1;
    inline static int av_sync_type = AV_SYNC_AUDIO_MASTER;
    inline static int64_t start_time = AV_NOPTS_VALUE;
    inline static int64_t duration = AV_NOPTS_VALUE;
    inline static int fast = 0;
    inline static int genpts = 0;
    inline static int lowres = 0;
    inline static int decoder_reorder_pts = -1;
    inline static int autoexit;
    inline static int exit_on_keydown;
    inline static int exit_on_mousedown;
    inline static int loop = 1;
    inline static int framedrop = -1;
    inline static int infinite_buffer = -1;
    inline static enum ShowMode show_mode = SHOW_MODE_NONE;
    inline static const char* audio_codec_name;
    inline static const char* subtitle_codec_name;
    inline static const char* video_codec_name;
    inline static double rdftspeed = 0.02;
    inline static int64_t cursor_last_shown;
    inline static int cursor_hidden = 0;
#if CONFIG_AVFILTER
    inline static const char** vfilters_list = NULL;
    inline static int nb_vfilters = 0;
    inline static char* afilters = NULL;
#endif
    inline static int autorotate = 1;
    inline static int find_stream_info = 1;
    inline static int filter_nbthreads = 0;

    /* current context */
    inline static int is_full_screen;
    inline static int64_t audio_callback_time;


    inline static const OptionDef options[] = {
    CMDUTILS_COMMON_OPTIONS
    { "x", HAS_ARG, {.func_arg = opt_width }, "force displayed width", "width" },
    { "y", HAS_ARG, {.func_arg = opt_height }, "force displayed height", "height" },
    { "s", HAS_ARG | OPT_VIDEO, {.func_arg = opt_frame_size }, "set frame size (WxH or abbreviation)", "size" },
    { "fs", OPT_BOOL, { &is_full_screen }, "force full screen" },
    { "an", OPT_BOOL, { &audio_disable }, "disable audio" },
    { "vn", OPT_BOOL, { &video_disable }, "disable video" },
    { "sn", OPT_BOOL, { &subtitle_disable }, "disable subtitling" },
    { "ast", OPT_STRING | HAS_ARG | OPT_EXPERT, { &wanted_stream_spec[AVMEDIA_TYPE_AUDIO] }, "select desired audio stream", "stream_specifier" },
    { "vst", OPT_STRING | HAS_ARG | OPT_EXPERT, { &wanted_stream_spec[AVMEDIA_TYPE_VIDEO] }, "select desired video stream", "stream_specifier" },
    { "sst", OPT_STRING | HAS_ARG | OPT_EXPERT, { &wanted_stream_spec[AVMEDIA_TYPE_SUBTITLE] }, "select desired subtitle stream", "stream_specifier" },
    { "ss", HAS_ARG, {.func_arg = opt_seek }, "seek to a given position in seconds", "pos" },
    { "t", HAS_ARG, {.func_arg = opt_duration }, "play  \"duration\" seconds of audio/video", "duration" },
    { "bytes", OPT_INT | HAS_ARG, { &seek_by_bytes }, "seek by bytes 0=off 1=on -1=auto", "val" },
    { "seek_interval", OPT_FLOAT | HAS_ARG, { &seek_interval }, "set seek interval for left/right keys, in seconds", "seconds" },
    { "nodisp", OPT_BOOL, { &display_disable }, "disable graphical display" },
    { "noborder", OPT_BOOL, { &borderless }, "borderless window" },
    { "volume", OPT_INT | HAS_ARG, { &startup_volume}, "set startup volume 0=min 100=max", "volume" },
    { "f", HAS_ARG, {.func_arg = opt_format }, "force format", "fmt" },
    { "pix_fmt", HAS_ARG | OPT_EXPERT | OPT_VIDEO, {.func_arg = opt_frame_pix_fmt }, "set pixel format", "format" },
    { "stats", OPT_BOOL | OPT_EXPERT, { &show_status }, "show status", "" },
    { "fast", OPT_BOOL | OPT_EXPERT, { &fast }, "non spec compliant optimizations", "" },
    { "genpts", OPT_BOOL | OPT_EXPERT, { &genpts }, "generate pts", "" },
    { "drp", OPT_INT | HAS_ARG | OPT_EXPERT, { &decoder_reorder_pts }, "let decoder reorder pts 0=off 1=on -1=auto", ""},
    { "lowres", OPT_INT | HAS_ARG | OPT_EXPERT, { &lowres }, "", "" },
    { "sync", HAS_ARG | OPT_EXPERT, {.func_arg = opt_sync }, "set audio-video sync. type (type=audio/video/ext)", "type" },
    { "autoexit", OPT_BOOL | OPT_EXPERT, { &autoexit }, "exit at the end", "" },
    { "exitonkeydown", OPT_BOOL | OPT_EXPERT, { &exit_on_keydown }, "exit on key down", "" },
    { "exitonmousedown", OPT_BOOL | OPT_EXPERT, { &exit_on_mousedown }, "exit on mouse down", "" },
    { "loop", OPT_INT | HAS_ARG | OPT_EXPERT, { &loop }, "set number of times the playback shall be looped", "loop count" },
    { "framedrop", OPT_BOOL | OPT_EXPERT, { &framedrop }, "drop frames when cpu is too slow", "" },
    { "infbuf", OPT_BOOL | OPT_EXPERT, { &infinite_buffer }, "don't limit the input buffer size (useful with realtime streams)", "" },
    { "window_title", OPT_STRING | HAS_ARG, { &window_title }, "set window title", "window title" },
    { "left", OPT_INT | HAS_ARG | OPT_EXPERT, { &screen_left }, "set the x position for the left of the window", "x pos" },
    { "top", OPT_INT | HAS_ARG | OPT_EXPERT, { &screen_top }, "set the y position for the top of the window", "y pos" },
#if CONFIG_AVFILTER
    { "vf", OPT_EXPERT | HAS_ARG, {.func_arg = opt_add_vfilter }, "set video filters", "filter_graph" },
    { "af", OPT_STRING | HAS_ARG, { &afilters }, "set audio filters", "filter_graph" },
#endif
    //{ "rdftspeed", OPT_INT | HAS_ARG | OPT_AUDIO | OPT_EXPERT, { &rdftspeed }, "rdft speed", "msecs" },
    { "showmode", HAS_ARG, {.func_arg = opt_show_mode}, "select show mode (0 = video, 1 = waves, 2 = RDFT)", "mode" },
    { "default", HAS_ARG | OPT_AUDIO | OPT_VIDEO | OPT_EXPERT, {.func_arg = opt_default }, "generic catch all option", "" },
    { "i", OPT_BOOL, { &dummy}, "read specified file", "input_file"},
    { "codec", HAS_ARG, {.func_arg = opt_codec}, "force decoder", "decoder_name" },
    { "acodec", HAS_ARG | OPT_STRING | OPT_EXPERT, {    &audio_codec_name }, "force audio decoder",    "decoder_name" },
    { "scodec", HAS_ARG | OPT_STRING | OPT_EXPERT, { &subtitle_codec_name }, "force subtitle decoder", "decoder_name" },
    { "vcodec", HAS_ARG | OPT_STRING | OPT_EXPERT, {    &video_codec_name }, "force video decoder",    "decoder_name" },
    { "autorotate", OPT_BOOL, { &autorotate }, "automatically rotate video", "" },
    { "find_stream_info", OPT_BOOL | OPT_INPUT | OPT_EXPERT, { &find_stream_info },
        "read and decode the streams to fill missing information with heuristics" },
    { "filter_threads", HAS_ARG | OPT_INT | OPT_EXPERT, { &filter_nbthreads }, "number of filter threads per graph" },
    { NULL, },
    };
};

