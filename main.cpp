#include "mainwindow.h"

extern "C" {
#include "cmdutils.h"
#include "libavformat/avformat.h"
#include "libavdevice/avdevice.h"
#include "libavutil/opt.h"
}

#include <SDL.h>
#include <SDL_thread.h>

#include <assert.h>

#include "CommandOptions.h"

#include "PacketQueue.h"
#include "FrameQueue.h"
#include "Decoder.h"
#include "Clock.h"
//#include "VideoState.h"
#include "Display.h"
//#include "EventHandler.h"
//#include "model.h"

#include <QApplication>

#include <cuda.h>
#include <cuda_runtime.h>

#pragma comment(lib, "user32.lib")
#pragma comment(lib, "Advapi32.lib")
#pragma comment(lib, "gdi32.lib")
#pragma comment(lib, "ole32.lib")
#pragma comment(lib, "oleaut32.lib")

#ifdef main
#undef main
#endif

const char program_name[] = "ffplay";
const int program_birth_year = 2003;

static CommandOptions co;

void show_help_default(const char* opt, const char* arg)
{
    av_log_set_callback(log_callback_help);
    co.show_usage();
    show_help_options(co.options, "Main options:", 0, OPT_EXPERT, 0);
    show_help_options(co.options, "Advanced options:", OPT_EXPERT, 0, 0);
    printf("\n");
    show_help_children(avcodec_get_class(), AV_OPT_FLAG_DECODING_PARAM);
    show_help_children(avformat_get_class(), AV_OPT_FLAG_DECODING_PARAM);
#if !CONFIG_AVFILTER
    show_help_children(sws_get_class(), AV_OPT_FLAG_ENCODING_PARAM);
#else
    show_help_children(avfilter_get_class(), AV_OPT_FLAG_FILTERING_PARAM);
#endif
    printf("\nWhile playing:\n"
        "q, ESC              quit\n"
        "f                   toggle full screen\n"
        "p, SPC              pause\n"
        "m                   toggle mute\n"
        "9, 0                decrease and increase volume respectively\n"
        "/, *                decrease and increase volume respectively\n"
        "a                   cycle audio channel in the current program\n"
        "v                   cycle video channel\n"
        "t                   cycle subtitle channel in the current program\n"
        "c                   cycle program\n"
        "w                   cycle video filters or show modes\n"
        "s                   activate frame-step mode\n"
        "left/right          seek backward/forward 10 seconds or to custom interval if -seek_interval is set\n"
        "down/up             seek backward/forward 1 minute\n"
        "page down/page up   seek backward/forward 10 minutes\n"
        "right mouse click   seek to percentage in file corresponding to fraction of width\n"
        "left double-click   toggle full screen\n"
    );
}

int main(int argc, char *argv[])
{
    init_dynload();
    av_log_set_flags(AV_LOG_SKIP_REPEATED);
    parse_loglevel(argc, argv, co.options);
#if CONFIG_AVDEVICE
    avdevice_register_all();
#endif
    avformat_network_init();
    init_opts();
    parse_options(NULL, argc, argv, co.options, co.opt_input_file);

    if (!co.input_filename) {
        //co.show_usage();
        co.input_filename = "C:\\Users\\sr996\\Videos\\odessa_0.mp4";
        //co.input_filename = "C:\\Users\\sr996\\Pictures\\test.jpg";
    }

    if (co.display_disable) {
        co.video_disable = 1;
    }
    int flags = SDL_INIT_VIDEO | SDL_INIT_AUDIO | SDL_INIT_TIMER;
    if (co.audio_disable)
        flags &= ~SDL_INIT_AUDIO;
    else {
        if (!SDL_getenv("SDL_AUDIO_ALSA_SET_BUFFER_SIZE"))
            SDL_setenv("SDL_AUDIO_ALSA_SET_BUFFER_SIZE", "1", 1);
    }

    if (co.display_disable)
        flags &= ~SDL_INIT_VIDEO;

    if (SDL_Init(flags)) {
        av_log(NULL, AV_LOG_FATAL, "Could not initialize SDL - %s\n", SDL_GetError());
        av_log(NULL, AV_LOG_FATAL, "(Did you set the DISPLAY variable?)\n");
        exit(1);
    }

    SDL_EventState(SDL_SYSWMEVENT, SDL_IGNORE);
    SDL_EventState(SDL_USEREVENT, SDL_IGNORE);

    QApplication a(argc, argv);
    MainWindow w;
    w.co = &co;
    w.initializeSDL();
    w.show();
    int result = a.exec();
    uninit_opts();
#if CONFIG_AVFILTER
    av_freep(&co.vfilters_list);
#endif
    avformat_network_deinit();
    if (co.show_status)
        printf("\n");
    SDL_Quit();

    return result;
}
