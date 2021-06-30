#pragma once

extern"C" {
#include "cmdutils.h"
#include "libavutil/opt.h"
}

#include <SDL.h>
#include <QObject>
#include <QString>
#include <QMainWindow>

//#include "VideoState.h"

//#define CO ((CommandOptions*)command_options)

#define NUM_OPTIONS 75

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

class CommandOptions : public QObject
{
    Q_OBJECT

public:
    CommandOptions();

    void showHelpCallback(void *ptr, int level, const char *fmt, va_list vl);
    void show_log_level(int log_level);
    int findOptionIndexByHelp(QString help);
    int findOptionIndexByName(QString name);
    bool av_log_on = false;

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
    static int opt_add_vfilter(void* optctx, const char* opt, const char* arg);

    inline static QMainWindow *mainWindow;

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
    inline static const char** vfilters_list = NULL;
    inline static int nb_vfilters = 0;
    inline static char* afilters = NULL;
    inline static int autorotate = 1;
    inline static int find_stream_info = 1;
    inline static int filter_nbthreads = 0;

    inline static const char* clock_sync;
    inline static const char* forced_format;

    /* current context */
    inline static int is_full_screen;
    inline static int64_t audio_callback_time;


    inline static OptionDef options[NUM_OPTIONS];

signals:
    void showHelp(const QString&);

};

