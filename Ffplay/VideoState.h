#pragma once

#include <iostream>

extern "C" {
#include "libavcodec/avfft.h"
#include "libavfilter/avfilter.h"
#include "libswresample/swresample.h"
#include "libavutil/avstring.h"
#include "libavfilter/buffersink.h"
#include "libavfilter/buffersrc.h"
#include "libavutil/pixdesc.h"
}

#include <QMainWindow>
#include <QMutex>

#include "Clock.h"
#include "FrameQueue.h"
#include "Decoder.h"
#include "Display.h"
#include "CommandOptions.h"
#include "simplefilter.h"
#include "streampanel.h"
#include "config.h"
#include "Filters/filterchain.h"
#include "Utilities/avexception.h"

#define AV_SYNC_THRESHOLD_MIN 0.04
#define AV_SYNC_THRESHOLD_MAX 0.1
#define AV_SYNC_FRAMEDUP_THRESHOLD 0.1
#define FF_QUIT_EVENT    (SDL_USEREVENT + 2)
#define FF_START_EVENT   (SDL_USEREVENT + 3)
#define MAX_QUEUE_SIZE (15 * 1024 * 1024)
#define MIN_FRAMES 25
#define VIDEO_PICTURE_QUEUE_SIZE 3
#define SUBPICTURE_QUEUE_SIZE 16
#define SAMPLE_QUEUE_SIZE 9
#define CURSOR_HIDE_DELAY 1000000
#define REFRESH_RATE 0.01
#define SDL_VOLUME_STEP (0.75)

typedef struct AudioParams {
    int freq;
    int channels;
    int64_t channel_layout;
    enum AVSampleFormat fmt;
    int frame_size;
    int bytes_per_sec;
} AudioParams;

class VideoState
{

public:
    VideoState();

    static VideoState* stream_open(QMainWindow *mw);

    void video_image_display();
    int compute_mod(int a, int b);
    int is_realtime(AVFormatContext* s);
    int stream_has_enough_packets(AVStream* st, int stream_id, PacketQueue* queue);
    int64_t get_valid_channel_layout(int64_t channel_layout, int channels);
    int cmp_audio_fmts(enum AVSampleFormat fmt1, int64_t channel_count1, enum AVSampleFormat fmt2, int64_t channel_count2);
    void stream_component_close(int stream_index);
    void stream_close();
    int get_master_sync_type();
    double get_master_clock();
    void check_external_clock_speed();
    void stream_seek(int64_t pos, int64_t rel, int seek_by_bytes);
    void seek_chapter(int inrc);
    void stream_toggle_pause();
    void toggle_pause();
    void toggle_mute();
    void toggle_full_screen();
    void toggle_audio_display();
    void update_volume(int sign, double step);
    void set_volume(double arg);
    void step_to_next_frame();
    void set_default_window_size(int width, int height, AVRational sar);
    int queue_picture(AVFrame* src_frame, double pts, double duration, int64_t pos, int serial);
    int get_video_frame(AVFrame* frame);
    double compute_target_delay(double delay);
    double vp_duration(Frame* vp, Frame* nextvp);
    void update_video_pts(double pts, int64_t pos, int serial);

    int video_open();
    void video_display();
    void update_sample_display(short* samples, int samples_size);
    void video_refresh(double* time_remaining);
    void subtitle_refresh();
    void show_status();

    int audio_thread();
    int video_thread();
    int subtitle_thread();
    int read_thread();
    void read_loop();
    void assign_read_options();
    
    int stream_component_open(int stream_index);

    void sdl_audio_callback(Uint8* stream, int len);
    int audio_open(int64_t wanted_channel_layout, int wanted_nb_channels, int wanted_sample_rate, struct AudioParams* audio_hw_params);

    int configure_filtergraph(AVFilterGraph* graph, const char* filtergraph, AVFilterContext* source_ctx, AVFilterContext* sink_ctx);
    int configure_video_filters(AVFilterGraph* graph, const char* vfilters, AVFrame* frame);
    int configure_audio_filters(const char* afilters, int force_output_format);

    int synchronize_audio(int nb_samples);
    int audio_decode_frame();
    void refresh_loop_wait_event(SDL_Event* event);
    void do_exit();

    void rewind();
    void fastforward();
    const QString formatTime(double time_in_seconds);
    static int readStreamData(void *opaque, uint8_t *buffer, int size);

    double elapsed;
    double total;
    double current_time;
    QString codec_name;

    QMainWindow* mainWindow;
    FilterChain* filterChain;
    SimpleFilter* filter;
    AVExceptionHandler av;
    SDL_mutex *display_mutex;
    AVIOContext *stream_ctx = nullptr;
    StreamData localData;
    uint8_t *stream_data;

    CommandOptions* co;
    Display* disp;
    AVPacket* flush_pkt;

    SDL_Thread* read_tid;
    AVInputFormat* iformat;
    AVFormatContext* ic;
    SDL_cond* continue_read_thread;

    int abort_request;
    int force_refresh;
    int paused;
    int last_paused;
    int queue_attachments_req;
    int seek_req;
    int seek_flags;
    int64_t seek_pos;
    int64_t seek_rel;
    int read_pause_return;
    int realtime;

    Clock audclk;
    Clock vidclk;
    Clock extclk;

    FrameQueue pictq;
    FrameQueue subpq;
    FrameQueue sampq;

    Decoder auddec;
    Decoder viddec;
    Decoder subdec;

    int audio_stream;

    int av_sync_type;

    double audio_clock;
    int audio_clock_serial;
    double audio_diff_cum; // used for AV difference average computation 
    double audio_diff_avg_coef;
    double audio_diff_threshold;
    int audio_diff_avg_count;
    AVStream* audio_st;
    PacketQueue audioq;
    int audio_hw_buf_size;
    uint8_t* audio_buf;
    uint8_t* audio_buf1;
    unsigned int audio_buf_size; // in bytes 
    unsigned int audio_buf1_size;
    int audio_buf_index; // in bytes 
    int audio_write_buf_size;
    int audio_volume;
    int muted;
    struct AudioParams audio_src;
    struct AudioParams audio_filter_src;
    struct AudioParams audio_tgt;
    struct SwrContext* swr_ctx;
    int frame_drops_early;
    int frame_drops_late;

    enum ShowMode show_mode;

    int16_t sample_array[SAMPLE_ARRAY_SIZE];
    int sample_array_index;
    int last_i_start;
    RDFTContext* rdft;
    int rdft_bits;
    FFTSample* rdft_data;
    int xpos;
    double last_vis_time;
    SDL_Texture* vis_texture;
    SDL_Texture* sub_texture;
    SDL_Texture* vid_texture;

    int subtitle_stream;
    AVStream* subtitle_st;
    PacketQueue subtitleq;

    double frame_timer;
    double frame_last_returned_time;
    double frame_last_filter_delay;
    double codec_frame_duration;
    int video_stream = -1;
    AVStream* video_st;
    PacketQueue videoq;
    double max_frame_duration;      // maximum duration of a frame - above this, we consider the jump a timestamp discontinuity
    struct SwsContext* img_convert_ctx;
    struct SwsContext* sub_convert_ctx;
    int eof;

    char* filename;
    int width, height, xleft, ytop;
    int step;

    int vfilter_idx;
    AVFilterContext* in_video_filter;   // the first filter in the video chain
    AVFilterContext* out_video_filter;  // the last filter in the video chain
    AVFilterContext* in_audio_filter;   // the first filter in the audio chain
    AVFilterContext* out_audio_filter;  // the last filter in the audio chain
    AVFilterGraph* agraph;              // audio filter graph

    //int last_video_stream, last_audio_stream, last_subtitle_stream;

};

