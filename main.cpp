#include "mainwindow.h"

extern "C" {
#include "Ffplay/cmdutils.h"
//#include "libavformat/avformat.h"
#include "libavdevice/avdevice.h"
#include "libavutil/opt.h"
#include "libavutil/ffversion.h"
}

#include <SDL.h>
//#include <SDL_thread.h>

//#include <assert.h>
//#include <functional>
#include "Ffplay/CommandOptions.h"

//#include "Ffplay/PacketQueue.h"
//#include "Ffplay/FrameQueue.h"
//#include "Ffplay/Decoder.h"
//#include "Ffplay/Clock.h"
//#include "VideoState.h"
//#include "Ffplay/Display.h"
//#include "EventHandler.h"
//#include "model.h"

#include <QApplication>

//#include <cuda.h>
//#include <cuda_runtime.h>

#pragma comment(lib, "user32.lib")
#pragma comment(lib, "Advapi32.lib")
#pragma comment(lib, "gdi32.lib")
#pragma comment(lib, "ole32.lib")
#pragma comment(lib, "oleaut32.lib")

#ifdef main
#undef main
#endif

#define INDENT         1
#define SHOW_VERSION   2
#define SHOW_CONFIG    4
#define SHOW_COPYRIGHT 8

const char program_name[] = "ffplay";
const int program_birth_year = 2003;

CommandOptions co;

void logCallback(void *ptr, int level, const char *fmt, va_list vl)
{
    co.showHelpCallback(ptr, level, fmt, vl);
}

void show_program_configs()
{
    co.av_log_on = true;
    av_log_set_callback(logCallback);

    /*
    int flags = INDENT|0;
    int level = AV_LOG_INFO;

    print_all_libs_info(INDENT|SHOW_CONFIG, AV_LOG_INFO);
    const char *indent = flags & INDENT? "  " : "";
    av_log(NULL, level, "%s version " FFMPEG_VERSION, program_name);
    av_log(NULL, level, " Copyright (c) %d-%d the FFmpeg developers",
           program_birth_year, CONFIG_THIS_YEAR);
    av_log(NULL, level, "\n");
    av_log(NULL, level, "%sbuilt with %s\n", indent, CC_IDENT);

    print_buildconf(INDENT|0, AV_LOG_INFO);
    print_all_libs_info(INDENT|SHOW_VERSION, AV_LOG_INFO);
    */

    co.av_log_on = false;
}

void show_help_default(const char* opt, const char* arg)
{
    co.av_log_on = true;
    av_log_set_callback(logCallback);
    co.show_usage();
    show_help_children(avcodec_get_class(), AV_OPT_FLAG_DECODING_PARAM);
    co.av_log_on = false;
    av_log_set_callback(log_callback_help);

}

int main(int argc, char *argv[])
{
    init_dynload();
    av_log_set_flags(AV_LOG_SKIP_REPEATED);
    parse_loglevel(argc, argv, co.options);
    avdevice_register_all();
    avformat_network_init();
    init_opts();
    parse_options(NULL, argc, argv, co.options, co.opt_input_file);

    int flags = SDL_INIT_VIDEO | SDL_INIT_AUDIO | SDL_INIT_TIMER;
    if (SDL_Init(flags)) {
        av_log(NULL, AV_LOG_FATAL, "Could not initialize SDL - %s\n", SDL_GetError());
        av_log(NULL, AV_LOG_FATAL, "(Did you set the DISPLAY variable?)\n");
        exit(1);
    }

    QApplication a(argc, argv);
    MainWindow w(&co);
    w.initializeSDL();
    w.show();
    int result = a.exec();

    uninit_opts();
    av_freep(&co.vfilters_list);
    avformat_network_deinit();
    SDL_Quit();

    return result;
}
