#include "VideoState.h"
#include "mainwindow.h"

VideoState::VideoState()
{
	memset(this, 0, sizeof(VideoState));
}

void VideoState::video_image_display()
{
    Frame* vp;
    Frame* sp = NULL;
    SDL_Rect rect;

    vp = pictq.peek_last();
    if (subtitle_st) {
        if (subpq.nb_remaining() > 0) {
            sp = subpq.peek();

            if (vp->pts >= sp->pts + ((float)sp->sub.start_display_time / 1000)) {
                if (!sp->uploaded) {
                    uint8_t* pixels[4];
                    int pitch[4];
                    int i;
                    if (!sp->width || !sp->height) {
                        sp->width = vp->width;
                        sp->height = vp->height;
                    }
                    if (disp->realloc_texture(&sub_texture, SDL_PIXELFORMAT_ARGB8888, sp->width, sp->height, SDL_BLENDMODE_BLEND, 1) < 0)
                        return;

                    for (i = 0; i < sp->sub.num_rects; i++) {
                        AVSubtitleRect* sub_rect = sp->sub.rects[i];

                        sub_rect->x = av_clip(sub_rect->x, 0, sp->width);
                        sub_rect->y = av_clip(sub_rect->y, 0, sp->height);
                        sub_rect->w = av_clip(sub_rect->w, 0, sp->width - sub_rect->x);
                        sub_rect->h = av_clip(sub_rect->h, 0, sp->height - sub_rect->y);

                        sub_convert_ctx = sws_getCachedContext(sub_convert_ctx,
                            sub_rect->w, sub_rect->h, AV_PIX_FMT_PAL8,
                            sub_rect->w, sub_rect->h, AV_PIX_FMT_BGRA,
                            0, NULL, NULL, NULL);
                        if (!sub_convert_ctx) {
                            av_log(NULL, AV_LOG_FATAL, "Cannot initialize the conversion context\n");
                            return;
                        }
                        if (!SDL_LockTexture(sub_texture, (SDL_Rect*)sub_rect, (void**)pixels, pitch)) {
                            sws_scale(sub_convert_ctx, (const uint8_t* const*)sub_rect->data, sub_rect->linesize,
                                0, sub_rect->h, pixels, pitch);
                            SDL_UnlockTexture(sub_texture);
                        }
                    }
                    sp->uploaded = 1;
                }
            }
            else
                sp = NULL;
        }
    }

    filter->process(vp);  //  hook into playqt filtering system
    disp->calculate_display_rect(&rect, xleft, ytop, width, height, vp->width, vp->height, vp->sar);

    if (!vp->uploaded) {
        if (disp->upload_texture(&vid_texture, vp->frame, &img_convert_ctx) < 0)
            return;
        vp->uploaded = 1;
        vp->flip_v = vp->frame->linesize[0] < 0;
    }

    disp->set_sdl_yuv_conversion_mode(vp->frame);
    SDL_RenderCopyEx(disp->renderer, vid_texture, NULL, &rect, 0, NULL, (SDL_RendererFlip)(vp->flip_v ? SDL_FLIP_VERTICAL : 0));
    disp->set_sdl_yuv_conversion_mode(NULL);
    if (sp) {
#if USE_ONEPASS_SUBTITLE_RENDER
        SDL_RenderCopy(disp->renderer, sub_texture, NULL, &rect);
#else
        int i;
        double xratio = (double)rect.w / (double)sp->width;
        double yratio = (double)rect.h / (double)sp->height;
        for (i = 0; i < sp->sub.num_rects; i++) {
            SDL_Rect* sub_rect = (SDL_Rect*)sp->sub.rects[i];
            SDL_Rect target = { .x = rect.x + sub_rect->x * xratio,
                               .y = rect.y + sub_rect->y * yratio,
                               .w = sub_rect->w * xratio,
                               .h = sub_rect->h * yratio };
            SDL_RenderCopy(renderer, sub_texture, sub_rect, &target);
        }
#endif
    }
}

int VideoState::compute_mod(int a, int b)
{
    return a < 0 ? a % b + b : a % b;
}

int64_t VideoState::get_valid_channel_layout(int64_t channel_layout, int channels)
{
    if (channel_layout && av_get_channel_layout_nb_channels(channel_layout) == channels)
        return channel_layout;
    else
        return 0;
}

int VideoState::cmp_audio_fmts(enum AVSampleFormat fmt1, int64_t channel_count1, enum AVSampleFormat fmt2, int64_t channel_count2)
{
    // If channel count == 1, planar and non-planar formats are the same 
    if (channel_count1 == 1 && channel_count2 == 1)
        return av_get_packed_sample_fmt(fmt1) != av_get_packed_sample_fmt(fmt2);
    else
        return channel_count1 != channel_count2 || fmt1 != fmt2;
}
void VideoState::video_audio_display()
{
    int i, i_start, x, y1, y, ys, delay, n, nb_display_channels;
    int ch, channels, h, h2;
    int64_t time_diff;
    int rdft_bits, nb_freq;

    for (rdft_bits = 1; (1 << rdft_bits) < 2 * height; rdft_bits++)
        ;
    nb_freq = 1 << (rdft_bits - 1);

    // compute display index : center on currently output samples 
    channels = audio_tgt.channels;
    nb_display_channels = channels;
    if (!paused) {
        int data_used = show_mode == (2 * nb_freq);
        n = 2 * channels;
        delay = audio_write_buf_size;
        delay /= n;

        // to be more precise, we take into account the time spent since
        //   the last buffer computation 
        if (co->audio_callback_time) {
            time_diff = av_gettime_relative() - co->audio_callback_time;
            delay -= (time_diff * audio_tgt.freq) / 1000000;
        }

        delay += 2 * data_used;
        if (delay < data_used)
            delay = data_used;

        i_start = x = compute_mod(sample_array_index - delay * channels, SAMPLE_ARRAY_SIZE);
        last_i_start = i_start;
    }
    else {
        i_start = last_i_start;
    }

    if (disp->realloc_texture(&vis_texture, SDL_PIXELFORMAT_ARGB8888, width, height, SDL_BLENDMODE_NONE, 1) < 0)
        return;

    nb_display_channels = FFMIN(nb_display_channels, 2);
    if (rdft_bits != rdft_bits) {
        av_rdft_end(rdft);
        av_free(rdft_data);
        rdft = av_rdft_init(rdft_bits, DFT_R2C);
        rdft_bits = rdft_bits;
        rdft_data = (FFTSample*)av_malloc_array(nb_freq, 4 * sizeof(*rdft_data));
    }
    if (!rdft || !rdft_data) {
        av_log(NULL, AV_LOG_ERROR, "Failed to allocate buffers for RDFT");
        std::cout << "Old SHOW_WAVES stuff" << std::endl;
        return;
    }
    else {
        FFTSample* data[2];
        //SDL_Rect rect = { .x = s->xpos, .y = 0, .w = 1, .h = s->height };
        SDL_Rect rect = { xpos, 0, 1, height };
        uint32_t* pixels;
        int pitch;
        for (ch = 0; ch < nb_display_channels; ch++) {
            data[ch] = rdft_data + 2 * nb_freq * ch;
            i = i_start + ch;
            for (x = 0; x < 2 * nb_freq; x++) {
                double w = (x - nb_freq) * (1.0 / nb_freq);
                data[ch][x] = sample_array[i] * (1.0 - w * w);
                i += channels;
                if (i >= SAMPLE_ARRAY_SIZE)
                    i -= SAMPLE_ARRAY_SIZE;
            }
            av_rdft_calc(rdft, data[ch]);
        }
        // Least efficient way to do this, we should of course
        // directly access it but it is more than fast enough. 
        if (!SDL_LockTexture(vis_texture, &rect, (void**)&pixels, &pitch)) {
            pitch >>= 2;
            pixels += pitch * height;
            for (y = 0; y < height; y++) {
                double w = 1 / sqrt(nb_freq);
                int a = sqrt(w * sqrt(data[0][2 * y + 0] * data[0][2 * y + 0] + data[0][2 * y + 1] * data[0][2 * y + 1]));
                int b = (nb_display_channels == 2) ? sqrt(w * hypot(data[1][2 * y + 0], data[1][2 * y + 1]))
                    : a;
                a = FFMIN(a, 255);
                b = FFMIN(b, 255);
                pixels -= pitch;
                *pixels = (a << 16) + (b << 8) + ((a + b) >> 1);
            }
            SDL_UnlockTexture(vis_texture);
        }
        SDL_RenderCopy(disp->renderer, vis_texture, NULL, NULL);
    }
    if (!paused)
        xpos++;
    if (xpos >= width)
        xpos = xleft;
}

void VideoState::stream_component_close(int stream_index)
{
    //AVFormatContext* ic = ic;
    AVCodecParameters* codecpar;

    if (stream_index < 0 || stream_index >= ic->nb_streams)
        return;
    codecpar = ic->streams[stream_index]->codecpar;

    switch (codecpar->codec_type) {
    case AVMEDIA_TYPE_AUDIO:
        auddec.abort(&sampq);
        SDL_CloseAudioDevice(disp->audio_dev);
        auddec.destroy();
        swr_free(&swr_ctx);
        av_freep(&audio_buf1);
        audio_buf1_size = 0;
        audio_buf = NULL;

        if (rdft) {
            av_rdft_end(rdft);
            av_freep(&rdft_data);
            rdft = NULL;
            rdft_bits = 0;
        }
        break;
    case AVMEDIA_TYPE_VIDEO:
        viddec.abort(&pictq);
        viddec.destroy();
        break;
    case AVMEDIA_TYPE_SUBTITLE:
        subdec.abort(&subpq);
        subdec.destroy();
        break;
    default:
        break;
    }

    ic->streams[stream_index]->discard = AVDISCARD_ALL;
    switch (codecpar->codec_type) {
    case AVMEDIA_TYPE_AUDIO:
        audio_st = NULL;
        audio_stream = -1;
        break;
    case AVMEDIA_TYPE_VIDEO:
        video_st = NULL;
        video_stream = -1;
        break;
    case AVMEDIA_TYPE_SUBTITLE:
        subtitle_st = NULL;
        subtitle_stream = -1;
        break;
    default:
        break;
    }
}

void VideoState::stream_close()
{
    // XXX: use a special url_shutdown call to abort parse cleanly 
    abort_request = 1;
    SDL_WaitThread(read_tid, NULL);

    // close each stream 
    if (audio_stream >= 0)
        stream_component_close(audio_stream);
    if (video_stream >= 0)
        stream_component_close(video_stream);
    if (subtitle_stream >= 0)
        stream_component_close(subtitle_stream);

    avformat_close_input(&ic);

    videoq.destroy();
    audioq.destroy();
    subtitleq.destroy();

    // free all pictures 
    pictq.destroy();
    sampq.destroy();
    subpq.destroy();
    SDL_DestroyCond(continue_read_thread);
    sws_freeContext(img_convert_ctx);
    sws_freeContext(sub_convert_ctx);
    av_free(filename);
    if (vis_texture)
        SDL_DestroyTexture(vis_texture);
    if (vid_texture)
        SDL_DestroyTexture(vid_texture);
    if (sub_texture)
        SDL_DestroyTexture(sub_texture);

    av_free(this);
}

int VideoState::get_master_sync_type() {
    if (av_sync_type == AV_SYNC_VIDEO_MASTER) {
        if (video_st)
            return AV_SYNC_VIDEO_MASTER;
        else
            return AV_SYNC_AUDIO_MASTER;
    }
    else if (av_sync_type == AV_SYNC_AUDIO_MASTER) {
        if (audio_st)
            return AV_SYNC_AUDIO_MASTER;
        else
            return AV_SYNC_EXTERNAL_CLOCK;
    }
    else {
        return AV_SYNC_EXTERNAL_CLOCK;
    }
}

double VideoState::get_master_clock()
{
    double val;

    switch (get_master_sync_type()) {
    case AV_SYNC_VIDEO_MASTER:
        val = vidclk.get_clock();
        break;
    case AV_SYNC_AUDIO_MASTER:
        val = audclk.get_clock();
        break;
    default:
        val = extclk.get_clock();
        break;
    }
    return val;
}

void VideoState::check_external_clock_speed() {
    if (video_stream >= 0 && videoq.nb_packets <= EXTERNAL_CLOCK_MIN_FRAMES ||
        audio_stream >= 0 && audioq.nb_packets <= EXTERNAL_CLOCK_MIN_FRAMES) {
        extclk.set_clock_speed(FFMAX(EXTERNAL_CLOCK_SPEED_MIN, extclk.speed - EXTERNAL_CLOCK_SPEED_STEP));
    }
    else if ((video_stream < 0 || videoq.nb_packets > EXTERNAL_CLOCK_MAX_FRAMES) &&
        (audio_stream < 0 || audioq.nb_packets > EXTERNAL_CLOCK_MAX_FRAMES)) {
        extclk.set_clock_speed(FFMIN(EXTERNAL_CLOCK_SPEED_MAX, extclk.speed + EXTERNAL_CLOCK_SPEED_STEP));
    }
    else {
        double speed = extclk.speed;
        if (speed != 1.0)
            extclk.set_clock_speed(speed + EXTERNAL_CLOCK_SPEED_STEP * (1.0 - speed) / fabs(1.0 - speed));
    }
}

void VideoState::stream_seek(int64_t pos, int64_t rel, int seek_by_bytes)
{
    if (!seek_req) {
        seek_pos = pos;
        seek_rel = rel;
        seek_flags &= ~AVSEEK_FLAG_BYTE;
        if (seek_by_bytes)
            seek_flags |= AVSEEK_FLAG_BYTE;
        seek_req = 1;
        SDL_CondSignal(continue_read_thread);
    }
}

void VideoState::seek_chapter(int incr)
{
    int64_t pos = get_master_clock() * AV_TIME_BASE;
    int i;

    if (!ic->nb_chapters)
        return;

    /* find the current chapter */
    for (i = 0; i < ic->nb_chapters; i++) {
        AVChapter* ch = ic->chapters[i];
        if (av_compare_ts(pos, av_make_q(1, AV_TIME_BASE), ch->start, ch->time_base) < 0) {
            i--;
            break;
        }
    }

    i += incr;
    i = FFMAX(i, 0);
    if (i >= ic->nb_chapters)
        return;

    av_log(NULL, AV_LOG_VERBOSE, "Seeking to chapter %d.\n", i);
    stream_seek(av_rescale_q(ic->chapters[i]->start, ic->chapters[i]->time_base,
        av_make_q(1, AV_TIME_BASE)), 0, 0);
}

void VideoState::stream_toggle_pause()
{
    if (paused) {
        frame_timer += av_gettime_relative() / 1000000.0 - vidclk.last_updated;
        if (read_pause_return != AVERROR(ENOSYS)) {
            vidclk.paused = 0;
        }
        vidclk.set_clock(vidclk.get_clock(), vidclk.serial);
    }
    extclk.set_clock(extclk.get_clock(), extclk.serial);
    paused = audclk.paused = vidclk.paused = extclk.paused = !paused;
}

void VideoState::toggle_pause()
{
    stream_toggle_pause();
    step = 0;
}

void VideoState::toggle_mute()
{
    muted = !muted;
}

void VideoState::toggle_full_screen()
{
    co->is_full_screen = !co->is_full_screen;
    SDL_SetWindowFullscreen(disp->window, co->is_full_screen ? SDL_WINDOW_FULLSCREEN_DESKTOP : 0);
}

void VideoState::toggle_audio_display()
{
    int next = show_mode;
    do {
        next = (next + 1) % SHOW_MODE_NB;
    } while (next != show_mode && (next == SHOW_MODE_VIDEO && !video_st || next != SHOW_MODE_VIDEO && !audio_st));
    if (show_mode != next) {
        force_refresh = 1;
        show_mode = (ShowMode)next;
    }
}

void VideoState::update_volume(int sign, double step)
{
    double volume_level = audio_volume ? (20 * log(audio_volume / (double)SDL_MIX_MAXVOLUME) / log(10)) : -1000.0;
    int new_volume = lrint(SDL_MIX_MAXVOLUME * pow(10.0, (volume_level + sign * step) / 20.0));
    audio_volume = av_clip(audio_volume == new_volume ? (audio_volume + sign) : new_volume, 0, SDL_MIX_MAXVOLUME);
}

void VideoState::step_to_next_frame()
{
    if (paused)
        stream_toggle_pause();
    step = 1;
}

void VideoState::set_default_window_size(int width, int height, AVRational sar)
{
    SDL_Rect rect;
    int max_width = co->screen_width ? co->screen_width : INT_MAX;
    int max_height = co->screen_height ? co->screen_height : INT_MAX;
    if (max_width == INT_MAX && max_height == INT_MAX)
        max_height = height;
    disp->calculate_display_rect(&rect, 0, 0, max_width, max_height, width, height, sar);
    co->default_width = rect.w;
    co->default_height = rect.h;
}

int VideoState::queue_picture(AVFrame* src_frame, double pts, double duration, int64_t pos, int serial)
{
    Frame* vp;

#if defined(DEBUG_SYNC)
    printf("frame_type=%c pts=%0.3f\n",
        av_get_picture_type_char(src_frame->pict_type), pts);
#endif

    if (!(vp = pictq.peek_writable()))
        return -1;

    vp->sar = src_frame->sample_aspect_ratio;
    vp->uploaded = 0;

    vp->width = src_frame->width;
    vp->height = src_frame->height;
    vp->format = src_frame->format;

    vp->pts = pts;
    vp->duration = duration;
    vp->pos = pos;
    vp->serial = serial;

    set_default_window_size(vp->width, vp->height, vp->sar);

    av_frame_move_ref(vp->frame, src_frame);
    pictq.push();
    return 0;
}

int VideoState::get_video_frame(AVFrame* frame)
{
    int got_picture;

    if ((got_picture = viddec.decode_frame(frame, NULL)) < 0)
        return -1;

    if (got_picture) {
        double dpts = NAN;

        if (frame->pts != AV_NOPTS_VALUE)
            dpts = av_q2d(video_st->time_base) * frame->pts;

        frame->sample_aspect_ratio = av_guess_sample_aspect_ratio(ic, video_st, frame);

        if (co->framedrop > 0 || (co->framedrop && get_master_sync_type() != AV_SYNC_VIDEO_MASTER)) {
            if (frame->pts != AV_NOPTS_VALUE) {
                double diff = dpts - get_master_clock();
                if (!isnan(diff) && fabs(diff) < AV_NOSYNC_THRESHOLD &&
                    diff - frame_last_filter_delay < 0 &&
                    viddec.pkt_serial == vidclk.serial &&
                    videoq.nb_packets) {
                    frame_drops_early++;
                    av_frame_unref(frame);
                    got_picture = 0;
                }
            }
        }
    }

    return got_picture;
}

double VideoState::compute_target_delay(double delay)
{
    double sync_threshold, diff = 0;

    // update delay to follow master synchronisation source 
    if (get_master_sync_type() != AV_SYNC_VIDEO_MASTER) {
        // if video is slave, we try to correct big delays by
        // duplicating or deleting a frame 
        diff = vidclk.get_clock() - get_master_clock();

        // skip or repeat frame. We take into account the
        // delay to compute the threshold. I still don't know
        // if it is the best guess 
        sync_threshold = FFMAX(AV_SYNC_THRESHOLD_MIN, FFMIN(AV_SYNC_THRESHOLD_MAX, delay));
        if (!isnan(diff) && fabs(diff) < max_frame_duration) {
            if (diff <= -sync_threshold)
                delay = FFMAX(0, delay + diff);
            else if (diff >= sync_threshold && delay > AV_SYNC_FRAMEDUP_THRESHOLD)
                delay = delay + diff;
            else if (diff >= sync_threshold)
                delay = 2 * delay;
        }
    }

    av_log(NULL, AV_LOG_TRACE, "video: delay=%0.3f A-V=%f\n",
        delay, -diff);

    return delay;
}

double VideoState::vp_duration(Frame* vp, Frame* nextvp) {
    if (vp->serial == nextvp->serial) {
        double duration = nextvp->pts - vp->pts;
        if (isnan(duration) || duration <= 0 || duration >max_frame_duration)
            return vp->duration;
        else
            return duration;
    }
    else {
        return 0.0;
    }
}

void VideoState::update_video_pts(double pts, int64_t pos, int serial) {
    vidclk.set_clock(pts, serial);
    extclk.sync_clock_to_slave(&vidclk);
}

void VideoState::stream_cycle_channel(int codec_type)
{
    //AVFormatContext* ic = is->ic;
    int start_index, stream_index;
    int old_index;
    AVStream* st;
    AVProgram* p = NULL;
    int nb_streams = ic->nb_streams;

    if (codec_type == AVMEDIA_TYPE_VIDEO) {
        start_index = last_video_stream;
        old_index = video_stream;
    }
    else if (codec_type == AVMEDIA_TYPE_AUDIO) {
        start_index = last_audio_stream;
        old_index = audio_stream;
    }
    else {
        start_index = last_subtitle_stream;
        old_index = subtitle_stream;
    }
    stream_index = start_index;

    if (codec_type != AVMEDIA_TYPE_VIDEO && video_stream != -1) {
        p = av_find_program_from_stream(ic, NULL, video_stream);
        if (p) {
            nb_streams = p->nb_stream_indexes;
            for (start_index = 0; start_index < nb_streams; start_index++)
                if (p->stream_index[start_index] == stream_index)
                    break;
            if (start_index == nb_streams)
                start_index = -1;
            stream_index = start_index;
        }
    }

    for (;;) {
        if (++stream_index >= nb_streams)
        {
            if (codec_type == AVMEDIA_TYPE_SUBTITLE)
            {
                stream_index = -1;
                last_subtitle_stream = -1;
                goto the_end;
            }
            if (start_index == -1)
                return;
            stream_index = 0;
        }
        if (stream_index == start_index)
            return;
        st = ic->streams[p ? p->stream_index[stream_index] : stream_index];
        if (st->codecpar->codec_type == codec_type) {
            /* check that parameters are OK */
            switch (codec_type) {
            case AVMEDIA_TYPE_AUDIO:
                if (st->codecpar->sample_rate != 0 &&
                    st->codecpar->channels != 0)
                    goto the_end;
                break;
            case AVMEDIA_TYPE_VIDEO:
            case AVMEDIA_TYPE_SUBTITLE:
                goto the_end;
            default:
                break;
            }
        }
    }
the_end:
    if (p && stream_index != -1)
        stream_index = p->stream_index[stream_index];
    av_log(NULL, AV_LOG_INFO, "Switch %s stream from #%d to #%d\n",
        av_get_media_type_string((AVMediaType)codec_type),
        old_index,
        stream_index);

    stream_component_close(old_index);
    stream_component_open(stream_index);
}

int VideoState::video_open()
{
    int w, h;

    w = co->screen_width ? co->screen_width : co->default_width;
    h = co->screen_height ? co->screen_height : co->default_height;

    if (!co->window_title)
        co->window_title = co->input_filename;
    SDL_SetWindowTitle(disp->window, co->window_title);

    SDL_SetWindowSize(disp->window, w, h);
    SDL_SetWindowPosition(disp->window, co->screen_left, co->screen_top);
    if (co->is_full_screen)
        SDL_SetWindowFullscreen(disp->window, SDL_WINDOW_FULLSCREEN_DESKTOP);
    SDL_ShowWindow(disp->window);

    width = w;
    height = h;

    return 0;
}

void VideoState::video_display()
{
    if (!width)
        video_open();

    SDL_SetRenderDrawColor(disp->renderer, 0, 0, 0, 255);
    SDL_RenderClear(disp->renderer);
    if (audio_st && show_mode != SHOW_MODE_VIDEO)
        video_audio_display();
    else if (video_st)
        video_image_display();
    SDL_RenderPresent(disp->renderer);
}

#if CONFIG_AVFILTER
int VideoState::configure_filtergraph(AVFilterGraph* graph, const char* filtergraph, AVFilterContext* source_ctx, AVFilterContext* sink_ctx)
{
    int ret, i;
    int nb_filters = graph->nb_filters;
    AVFilterInOut* outputs = NULL, * inputs = NULL;

    if (filtergraph) {
        outputs = avfilter_inout_alloc();
        inputs = avfilter_inout_alloc();
        if (!outputs || !inputs) {
            ret = AVERROR(ENOMEM);
            goto fail;
        }

        outputs->name = av_strdup("in");
        outputs->filter_ctx = source_ctx;
        outputs->pad_idx = 0;
        outputs->next = NULL;

        inputs->name = av_strdup("out");
        inputs->filter_ctx = sink_ctx;
        inputs->pad_idx = 0;
        inputs->next = NULL;

        if ((ret = avfilter_graph_parse_ptr(graph, filtergraph, &inputs, &outputs, NULL)) < 0)
            goto fail;
    }
    else {
        if ((ret = avfilter_link(source_ctx, 0, sink_ctx, 0)) < 0)
            goto fail;
    }

    // Reorder the filters to ensure that inputs of the custom filters are merged first 
    for (i = 0; i < graph->nb_filters - nb_filters; i++)
        FFSWAP(AVFilterContext*, graph->filters[i], graph->filters[i + nb_filters]);

    ret = avfilter_graph_config(graph, NULL);
fail:
    avfilter_inout_free(&outputs);
    avfilter_inout_free(&inputs);
    return ret;
}

int VideoState::configure_video_filters(AVFilterGraph* graph, const char* vfilters, AVFrame* frame)
{
    enum AVPixelFormat pix_fmts[FF_ARRAY_ELEMS(sdl_texture_format_map)];
    char sws_flags_str[512] = "";
    char buffersrc_args[256];
    int ret;
    AVFilterContext* filt_src = NULL, * filt_out = NULL, * last_filter = NULL;
    AVCodecParameters* codecpar = video_st->codecpar;
    AVRational fr = av_guess_frame_rate(ic, video_st, NULL);
    AVDictionaryEntry* e = NULL;
    int nb_pix_fmts = 0;
    int i, j;

    for (i = 0; i < disp->renderer_info.num_texture_formats; i++) {
        for (j = 0; j < FF_ARRAY_ELEMS(sdl_texture_format_map) - 1; j++) {
            if (disp->renderer_info.texture_formats[i] == sdl_texture_format_map[j].texture_fmt) {
                pix_fmts[nb_pix_fmts++] = sdl_texture_format_map[j].format;
                break;
            }
        }
    }
    pix_fmts[nb_pix_fmts] = AV_PIX_FMT_NONE;

    while ((e = av_dict_get(sws_dict, "", e, AV_DICT_IGNORE_SUFFIX))) {
        if (!strcmp(e->key, "sws_flags")) {
            av_strlcatf(sws_flags_str, sizeof(sws_flags_str), "%s=%s:", "flags", e->value);
        }
        else
            av_strlcatf(sws_flags_str, sizeof(sws_flags_str), "%s=%s:", e->key, e->value);
    }
    if (strlen(sws_flags_str))
        sws_flags_str[strlen(sws_flags_str) - 1] = '\0';

    graph->scale_sws_opts = av_strdup(sws_flags_str);

    snprintf(buffersrc_args, sizeof(buffersrc_args),
        "video_size=%dx%d:pix_fmt=%d:time_base=%d/%d:pixel_aspect=%d/%d",
        frame->width, frame->height, frame->format,
        video_st->time_base.num, video_st->time_base.den,
        codecpar->sample_aspect_ratio.num, FFMAX(codecpar->sample_aspect_ratio.den, 1));
    if (fr.num && fr.den)
        av_strlcatf(buffersrc_args, sizeof(buffersrc_args), ":frame_rate=%d/%d", fr.num, fr.den);

    if ((ret = avfilter_graph_create_filter(&filt_src,
        avfilter_get_by_name("buffer"),
        "ffplay_buffer", buffersrc_args, NULL,
        graph)) < 0)
        goto fail;

    ret = avfilter_graph_create_filter(&filt_out,
        avfilter_get_by_name("buffersink"),
        "ffplay_buffersink", NULL, NULL, graph);
    if (ret < 0)
        goto fail;

    if ((ret = av_opt_set_int_list(filt_out, "pix_fmts", pix_fmts, AV_PIX_FMT_NONE, AV_OPT_SEARCH_CHILDREN)) < 0)
        goto fail;

    last_filter = filt_out;

    /* Note: this macro adds a filter before the lastly added filter, so the
     * processing order of the filters is in reverse */
#define INSERT_FILT(name, arg) do {                                          \
    AVFilterContext *filt_ctx;                                               \
                                                                             \
    ret = avfilter_graph_create_filter(&filt_ctx,                            \
                                       avfilter_get_by_name(name),           \
                                       "ffplay_" name, arg, NULL, graph);    \
    if (ret < 0)                                                             \
        goto fail;                                                           \
                                                                             \
    ret = avfilter_link(filt_ctx, 0, last_filter, 0);                        \
    if (ret < 0)                                                             \
        goto fail;                                                           \
                                                                             \
    last_filter = filt_ctx;                                                  \
} while (0)

    if (co->autorotate) {
        double theta = get_rotation(video_st);

        if (fabs(theta - 90) < 1.0) {
            INSERT_FILT("transpose", "clock");
        }
        else if (fabs(theta - 180) < 1.0) {
            INSERT_FILT("hflip", NULL);
            INSERT_FILT("vflip", NULL);
        }
        else if (fabs(theta - 270) < 1.0) {
            INSERT_FILT("transpose", "cclock");
        }
        else if (fabs(theta) > 1.0) {
            char rotate_buf[64];
            snprintf(rotate_buf, sizeof(rotate_buf), "%f*PI/180", theta);
            INSERT_FILT("rotate", rotate_buf);
        }
    }

    if ((ret = configure_filtergraph(graph, vfilters, filt_src, last_filter)) < 0)
        goto fail;

    in_video_filter = filt_src;
    out_video_filter = filt_out;

fail:
    return ret;
}

int VideoState::configure_audio_filters(const char* afilters, int force_output_format)
{
    static const enum AVSampleFormat sample_fmts[] = { AV_SAMPLE_FMT_S16, AV_SAMPLE_FMT_NONE };
    int sample_rates[2] = { 0, -1 };
    int64_t channel_layouts[2] = { 0, -1 };
    int channels[2] = { 0, -1 };
    AVFilterContext* filt_asrc = NULL, * filt_asink = NULL;
    char aresample_swr_opts[512] = "";
    AVDictionaryEntry* e = NULL;
    char asrc_args[256];
    int ret;

    avfilter_graph_free(&agraph);
    if (!(agraph = avfilter_graph_alloc()))
        return AVERROR(ENOMEM);
    agraph->nb_threads = co->filter_nbthreads;

    while ((e = av_dict_get(swr_opts, "", e, AV_DICT_IGNORE_SUFFIX)))
        av_strlcatf(aresample_swr_opts, sizeof(aresample_swr_opts), "%s=%s:", e->key, e->value);
    if (strlen(aresample_swr_opts))
        aresample_swr_opts[strlen(aresample_swr_opts) - 1] = '\0';
    av_opt_set(agraph, "aresample_swr_opts", aresample_swr_opts, 0);

    ret = snprintf(asrc_args, sizeof(asrc_args),
        "sample_rate=%d:sample_fmt=%s:channels=%d:time_base=%d/%d",
        audio_filter_src.freq, av_get_sample_fmt_name(audio_filter_src.fmt),
        audio_filter_src.channels,
        1, audio_filter_src.freq);
    if (audio_filter_src.channel_layout)
        snprintf(asrc_args + ret, sizeof(asrc_args) - ret,
            ":channel_layout=0x%I64x", audio_filter_src.channel_layout);

    ret = avfilter_graph_create_filter(&filt_asrc,
        avfilter_get_by_name("abuffer"), "ffplay_abuffer",
        asrc_args, NULL, agraph);
    if (ret < 0)
        goto end;


    ret = avfilter_graph_create_filter(&filt_asink,
        avfilter_get_by_name("abuffersink"), "ffplay_abuffersink",
        NULL, NULL, agraph);
    if (ret < 0)
        goto end;

    if ((ret = av_opt_set_int_list(filt_asink, "sample_fmts", sample_fmts, AV_SAMPLE_FMT_NONE, AV_OPT_SEARCH_CHILDREN)) < 0)
        goto end;
    if ((ret = av_opt_set_int(filt_asink, "all_channel_counts", 1, AV_OPT_SEARCH_CHILDREN)) < 0)
        goto end;

    if (force_output_format) {
        channel_layouts[0] = audio_tgt.channel_layout;
        channels[0] = audio_tgt.channels;
        sample_rates[0] = audio_tgt.freq;
        if ((ret = av_opt_set_int(filt_asink, "all_channel_counts", 0, AV_OPT_SEARCH_CHILDREN)) < 0)
            goto end;
        if ((ret = av_opt_set_int_list(filt_asink, "channel_layouts", channel_layouts, -1, AV_OPT_SEARCH_CHILDREN)) < 0)
            goto end;
        if ((ret = av_opt_set_int_list(filt_asink, "channel_counts", channels, -1, AV_OPT_SEARCH_CHILDREN)) < 0)
            goto end;
        if ((ret = av_opt_set_int_list(filt_asink, "sample_rates", sample_rates, -1, AV_OPT_SEARCH_CHILDREN)) < 0)
            goto end;
    }


    if ((ret = configure_filtergraph(agraph, afilters, filt_asrc, filt_asink)) < 0)
        goto end;

    in_audio_filter = filt_asrc;
    out_audio_filter = filt_asink;

end:
    if (ret < 0)
        avfilter_graph_free(&agraph);
    return ret;
}
#endif // CONFIG_AVFILTER
/*
*/

int VideoState::synchronize_audio(int nb_samples)
{
    int wanted_nb_samples = nb_samples;

    // if not master, then we try to remove or add samples to correct the clock 
    if (get_master_sync_type() != AV_SYNC_AUDIO_MASTER) {
        double diff, avg_diff;
        int min_nb_samples, max_nb_samples;

        diff = audclk.get_clock() - get_master_clock();

        if (!isnan(diff) && fabs(diff) < AV_NOSYNC_THRESHOLD) {
            audio_diff_cum = diff + audio_diff_avg_coef * audio_diff_cum;
            if (audio_diff_avg_count < AUDIO_DIFF_AVG_NB) {
                // not enough measures to have a correct estimate 
                audio_diff_avg_count++;
            }
            else {
                // estimate the A-V difference 
                avg_diff = audio_diff_cum * (1.0 - audio_diff_avg_coef);

                if (fabs(avg_diff) >= audio_diff_threshold) {
                    wanted_nb_samples = nb_samples + (int)(diff * audio_src.freq);
                    min_nb_samples = ((nb_samples * (100 - SAMPLE_CORRECTION_PERCENT_MAX) / 100));
                    max_nb_samples = ((nb_samples * (100 + SAMPLE_CORRECTION_PERCENT_MAX) / 100));
                    wanted_nb_samples = av_clip(wanted_nb_samples, min_nb_samples, max_nb_samples);
                }
                av_log(NULL, AV_LOG_TRACE, "diff=%f adiff=%f sample_diff=%d apts=%0.3f %f\n",
                    diff, avg_diff, wanted_nb_samples - nb_samples,
                    audio_clock, audio_diff_threshold);
            }
        }
        else {
            // too big difference : may be initial PTS errors, so
            //   reset A-V filter 
            audio_diff_avg_count = 0;
            audio_diff_cum = 0;
        }
    }

    return wanted_nb_samples;
}

int VideoState::audio_decode_frame()
{
    int data_size, resampled_data_size;
    int64_t dec_channel_layout;
    av_unused double audio_clock0;
    int wanted_nb_samples;
    Frame* af;

    if (paused)
        return -1;

    do {
#if defined(_WIN32)
        while (sampq.nb_remaining() == 0) {
            if ((av_gettime_relative() - co->audio_callback_time) > 1000000LL * audio_hw_buf_size / audio_tgt.bytes_per_sec / 2)
                return -1;
            av_usleep(1000);
        }
#endif
        if (!(af = sampq.peek_readable()))
            return -1;
        sampq.next();
    } while (af->serial != audioq.serial);

    data_size = av_samples_get_buffer_size(NULL, af->frame->channels,
        af->frame->nb_samples,
        (AVSampleFormat)af->frame->format, 1);

    dec_channel_layout =
        (af->frame->channel_layout && af->frame->channels == av_get_channel_layout_nb_channels(af->frame->channel_layout)) ?
        af->frame->channel_layout : av_get_default_channel_layout(af->frame->channels);
    wanted_nb_samples = synchronize_audio(af->frame->nb_samples);

    if (af->frame->format != audio_src.fmt ||
        dec_channel_layout != audio_src.channel_layout ||
        af->frame->sample_rate != audio_src.freq ||
        (wanted_nb_samples != af->frame->nb_samples && !swr_ctx)) {
        swr_free(&swr_ctx);
        swr_ctx = swr_alloc_set_opts(NULL,
            audio_tgt.channel_layout, audio_tgt.fmt, audio_tgt.freq,
            dec_channel_layout, (AVSampleFormat)af->frame->format, af->frame->sample_rate,
            0, NULL);
        if (!swr_ctx || swr_init(swr_ctx) < 0) {
            av_log(NULL, AV_LOG_ERROR,
                "Cannot create sample rate converter for conversion of %d Hz %s %d channels to %d Hz %s %d channels!\n",
                af->frame->sample_rate, av_get_sample_fmt_name((AVSampleFormat)af->frame->format), af->frame->channels,
                audio_tgt.freq, av_get_sample_fmt_name(audio_tgt.fmt), audio_tgt.channels);
            swr_free(&swr_ctx);
            return -1;
        }
        audio_src.channel_layout = dec_channel_layout;
        audio_src.channels = af->frame->channels;
        audio_src.freq = af->frame->sample_rate;
        audio_src.fmt = (AVSampleFormat)af->frame->format;
    }

    if (swr_ctx) {
        const uint8_t** in = (const uint8_t**)af->frame->extended_data;
        uint8_t** out = &audio_buf1;
        int out_count = (int64_t)wanted_nb_samples * audio_tgt.freq / af->frame->sample_rate + 256;
        int out_size = av_samples_get_buffer_size(NULL, audio_tgt.channels, out_count, audio_tgt.fmt, 0);
        int len2;
        if (out_size < 0) {
            av_log(NULL, AV_LOG_ERROR, "av_samples_get_buffer_size() failed\n");
            return -1;
        }
        if (wanted_nb_samples != af->frame->nb_samples) {
            if (swr_set_compensation(swr_ctx, (wanted_nb_samples - af->frame->nb_samples) * audio_tgt.freq / af->frame->sample_rate,
                wanted_nb_samples * audio_tgt.freq / af->frame->sample_rate) < 0) {
                av_log(NULL, AV_LOG_ERROR, "swr_set_compensation() failed\n");
                return -1;
            }
        }
        av_fast_malloc(&audio_buf1, &audio_buf1_size, out_size);
        if (!audio_buf1)
            return AVERROR(ENOMEM);
        len2 = swr_convert(swr_ctx, out, out_count, in, af->frame->nb_samples);
        if (len2 < 0) {
            av_log(NULL, AV_LOG_ERROR, "swr_convert() failed\n");
            return -1;
        }
        if (len2 == out_count) {
            av_log(NULL, AV_LOG_WARNING, "audio buffer is probably too small\n");
            if (swr_init(swr_ctx) < 0)
                swr_free(&swr_ctx);
        }
        audio_buf = audio_buf1;
        resampled_data_size = len2 * audio_tgt.channels * av_get_bytes_per_sample(audio_tgt.fmt);
    }
    else {
        audio_buf = af->frame->data[0];
        resampled_data_size = data_size;
    }

    audio_clock0 = audio_clock;
    // update the audio clock with the pts 
    if (!isnan(af->pts))
        audio_clock = af->pts + (double)af->frame->nb_samples / af->frame->sample_rate;
    else
        audio_clock = NAN;
    audio_clock_serial = af->serial;
#ifdef DEBUG
    {
        static double last_clock;
        printf("audio: delay=%0.3f clock=%0.3f clock0=%0.3f\n",
            audio_clock - last_clock,
            audio_clock, audio_clock0);
        last_clock = audio_clock;
    }
#endif
    return resampled_data_size;
}

void VideoState::update_sample_display(short* samples, int samples_size)
{
    int size, len;

    size = samples_size / sizeof(short);
    while (size > 0) {
        len = SAMPLE_ARRAY_SIZE - sample_array_index;
        if (len > size)
            len = size;
        memcpy(sample_array + sample_array_index, samples, len * sizeof(short));
        samples += len;
        sample_array_index += len;
        if (sample_array_index >= SAMPLE_ARRAY_SIZE)
            sample_array_index = 0;
        size -= len;
    }
}

void VideoState::video_refresh(double* remaining_time)
{
    //VideoState* is = (VideoState*)opaque;
    double time;

    Frame* sp, * sp2;

    if (!paused && get_master_sync_type() == AV_SYNC_EXTERNAL_CLOCK && realtime)
        check_external_clock_speed();

    if (!co->display_disable && show_mode != SHOW_MODE_VIDEO && audio_st) {
        time = av_gettime_relative() / 1000000.0;
        if (force_refresh || last_vis_time + co->rdftspeed < time) {
            video_display();
            last_vis_time = time;
        }
        *remaining_time = FFMIN(*remaining_time, last_vis_time + co->rdftspeed - time);

    }

    if (video_st) {
    retry:
        if (pictq.nb_remaining() == 0) {
            // nothing to do, no picture to display in the queue
        }
        else {
            double last_duration, duration, delay;
            Frame* vp, * lastvp;

            // dequeue the picture 
            lastvp = pictq.peek_last();
            vp = pictq.peek();

            if (vp->serial != videoq.serial) {
                pictq.next();
                goto retry;
            }

            if (lastvp->serial != vp->serial)
                frame_timer = av_gettime_relative() / 1000000.0;

            if (paused)
                goto display;

            // compute nominal last_duration 
            last_duration = vp_duration(lastvp, vp);
            delay = compute_target_delay(last_duration);

            time = av_gettime_relative() / 1000000.0;
            if (time < frame_timer + delay) {
                *remaining_time = FFMIN(frame_timer + delay - time, *remaining_time);
                goto display;
            }

            frame_timer += delay;
            if (delay > 0 && time - frame_timer > AV_SYNC_THRESHOLD_MAX)
                frame_timer = time;

            SDL_LockMutex(pictq.mutex);
            if (!isnan(vp->pts))
                update_video_pts(vp->pts, vp->pos, vp->serial);
            SDL_UnlockMutex(pictq.mutex);

            if (pictq.nb_remaining() > 1) {
                Frame* nextvp = pictq.peek_next();
                duration = vp_duration(vp, nextvp);
                if (!step && (co->framedrop > 0 || (co->framedrop && get_master_sync_type() != AV_SYNC_VIDEO_MASTER)) && time > frame_timer + duration) {
                    frame_drops_late++;
                    pictq.next();
                    goto retry;
                }
            }

            if (subtitle_st) {
                while (subpq.nb_remaining() > 0) {
                    sp = subpq.peek();

                    if (subpq.nb_remaining() > 1)
                        sp2 = subpq.peek_next();
                    else
                        sp2 = NULL;

                    if (sp->serial != subtitleq.serial
                        || (vidclk.pts > (sp->pts + ((float)sp->sub.end_display_time / 1000)))
                        || (sp2 && vidclk.pts > (sp2->pts + ((float)sp2->sub.start_display_time / 1000))))
                    {
                        if (sp->uploaded) {
                            int i;
                            for (i = 0; i < sp->sub.num_rects; i++) {
                                AVSubtitleRect* sub_rect = sp->sub.rects[i];
                                uint8_t* pixels;
                                int pitch, j;

                                if (!SDL_LockTexture(sub_texture, (SDL_Rect*)sub_rect, (void**)&pixels, &pitch)) {
                                    for (j = 0; j < sub_rect->h; j++, pixels += pitch)
                                        memset(pixels, 0, sub_rect->w << 2);
                                    SDL_UnlockTexture(sub_texture);
                                }
                            }
                        }
                        subpq.next();
                    }
                    else {
                        break;
                    }
                }
            }

            pictq.next();
            force_refresh = 1;

            if (step && !paused)
                stream_toggle_pause();
        }
    display:
        // display picture 
        if (!co->display_disable && force_refresh && show_mode == SHOW_MODE_VIDEO && pictq.rindex_shown)
            video_display();
    }
    force_refresh = 0;
    if (co->show_status) {
        static int64_t last_time;
        int64_t cur_time;
        int aqsize, vqsize, sqsize;
        double av_diff;

        cur_time = av_gettime_relative();
        if (!last_time || (cur_time - last_time) >= 30000) {
            aqsize = 0;
            vqsize = 0;
            sqsize = 0;
            if (audio_st)
                aqsize = audioq.size;
            if (video_st)
                vqsize = videoq.size;
            if (subtitle_st)
                sqsize = subtitleq.size;
            av_diff = 0;
            if (audio_st && video_st)
                av_diff = audclk.get_clock() - vidclk.get_clock();
            else if (video_st)
                av_diff = get_master_clock() - vidclk.get_clock();
            else if (audio_st)
                av_diff = get_master_clock() - audclk.get_clock();
            av_log(NULL, AV_LOG_INFO,
                "%7.2f %s:%7.3f fd=%4d aq=%5dKB vq=%5dKB sq=%5dB f=%I64d/%I64d   \r",
                get_master_clock(),
                (audio_st && video_st) ? "A-V" : (video_st ? "M-V" : (audio_st ? "M-A" : "   ")),
                av_diff,
                frame_drops_early + frame_drops_late,
                aqsize / 1024,
                vqsize / 1024,
                sqsize,
                video_st ? viddec.avctx->pts_correction_num_faulty_dts : 0,
                video_st ? viddec.avctx->pts_correction_num_faulty_pts : 0);
            fflush(stdout);
            last_time = cur_time;
        }
    }
}

static int audioThread(void* opaque)
{
    return static_cast<VideoState*>(opaque)->audio_thread();
}

int VideoState::audio_thread()
{
    //VideoState* is = (VideoState*)arg;
    AVFrame* frame = av_frame_alloc();
    Frame* af;
#if CONFIG_AVFILTER
    int last_serial = -1;
    int64_t dec_channel_layout;
    int reconfigure;
#endif
    int got_frame = 0;
    AVRational tb;
    int ret = 0;

    if (!frame)
        return AVERROR(ENOMEM);

    do {
        if ((got_frame = auddec.decode_frame(frame, NULL)) < 0)
            goto the_end;

        if (got_frame) {
            tb = av_make_q(1, frame->sample_rate);

#if CONFIG_AVFILTER
            dec_channel_layout = get_valid_channel_layout(frame->channel_layout, frame->channels);

            reconfigure =
                cmp_audio_fmts(audio_filter_src.fmt, audio_filter_src.channels,
                    (AVSampleFormat)frame->format, frame->channels) ||
                audio_filter_src.channel_layout != dec_channel_layout ||
                audio_filter_src.freq != frame->sample_rate ||
                auddec.pkt_serial != last_serial;

            if (reconfigure) {
                char buf1[1024], buf2[1024];
                av_get_channel_layout_string(buf1, sizeof(buf1), -1, audio_filter_src.channel_layout);
                av_get_channel_layout_string(buf2, sizeof(buf2), -1, dec_channel_layout);
                av_log(NULL, AV_LOG_DEBUG,
                    "Audio frame changed from rate:%d ch:%d fmt:%s layout:%s serial:%d to rate:%d ch:%d fmt:%s layout:%s serial:%d\n",
                    audio_filter_src.freq, audio_filter_src.channels, av_get_sample_fmt_name(audio_filter_src.fmt), buf1, last_serial,
                    frame->sample_rate, frame->channels, av_get_sample_fmt_name((AVSampleFormat)frame->format), buf2, auddec.pkt_serial);

                audio_filter_src.fmt = (AVSampleFormat)frame->format;
                audio_filter_src.channels = frame->channels;
                audio_filter_src.channel_layout = dec_channel_layout;
                audio_filter_src.freq = frame->sample_rate;
                last_serial = auddec.pkt_serial;

                if ((ret = configure_audio_filters(co->afilters, 1)) < 0)
                    goto the_end;
            }

            if ((ret = av_buffersrc_add_frame(in_audio_filter, frame)) < 0)
                goto the_end;

            while ((ret = av_buffersink_get_frame_flags(out_audio_filter, frame, 0)) >= 0) {
                tb = av_buffersink_get_time_base(out_audio_filter);
#endif
                if (!(af = sampq.peek_writable()))
                    goto the_end;

                af->pts = (frame->pts == AV_NOPTS_VALUE) ? NAN : frame->pts * av_q2d(tb);
                af->pos = frame->pkt_pos;
                af->serial = auddec.pkt_serial;
                af->duration = av_q2d(av_make_q(frame->nb_samples, frame->sample_rate));

                av_frame_move_ref(af->frame, frame);
                sampq.push();

#if CONFIG_AVFILTER
                if (audioq.serial != auddec.pkt_serial)
                    break;
            }
            if (ret == AVERROR_EOF)
                auddec.finished = auddec.pkt_serial;
#endif
        }
    } while (ret >= 0 || ret == AVERROR(EAGAIN) || ret == AVERROR_EOF);
the_end:
#if CONFIG_AVFILTER
    avfilter_graph_free(&agraph);
#endif
    av_frame_free(&frame);
    return ret;
}

static int videoThread(void* opaque)
{
    return static_cast<VideoState*>(opaque)->video_thread();
}

int VideoState::video_thread()
{
    //VideoState* is = (VideoState*)arg;
    AVFrame* frame = av_frame_alloc();
    double pts;
    double duration;
    int ret;
    AVRational tb = video_st->time_base;
    AVRational frame_rate = av_guess_frame_rate(ic, video_st, NULL);

#if CONFIG_AVFILTER
    AVFilterGraph* graph = NULL;
    AVFilterContext* filt_out = NULL, * filt_in = NULL;
    int last_w = 0;
    int last_h = 0;
    enum AVPixelFormat last_format = (AVPixelFormat)-2;
    int last_serial = -1;
    int last_vfilter_idx = 0;
#endif

    if (!frame)
        return AVERROR(ENOMEM);

    for (;;) {
        ret = get_video_frame(frame);
        if (ret < 0)
            goto the_end;
        if (!ret)
            continue;

#if CONFIG_AVFILTER
        if (last_w != frame->width
            || last_h != frame->height
            || last_format != frame->format
            || last_serial != viddec.pkt_serial
            || last_vfilter_idx != vfilter_idx) {
            av_log(NULL, AV_LOG_DEBUG,
                "Video frame changed from size:%dx%d format:%s serial:%d to size:%dx%d format:%s serial:%d\n",
                last_w, last_h,
                (const char*)av_x_if_null(av_get_pix_fmt_name(last_format), "none"), last_serial,
                frame->width, frame->height,
                (const char*)av_x_if_null(av_get_pix_fmt_name((AVPixelFormat)frame->format), "none"), viddec.pkt_serial);
            avfilter_graph_free(&graph);
            graph = avfilter_graph_alloc();
            if (!graph) {
                ret = AVERROR(ENOMEM);
                goto the_end;
            }
            graph->nb_threads = co->filter_nbthreads;
            if ((ret = configure_video_filters(graph, co->vfilters_list ? co->vfilters_list[vfilter_idx] : NULL, frame)) < 0) {
                SDL_Event event;
                event.type = FF_QUIT_EVENT;
                event.user.data1 = this;
                SDL_PushEvent(&event);
                goto the_end;
            }
            filt_in = in_video_filter;
            filt_out = out_video_filter;
            last_w = frame->width;
            last_h = frame->height;
            last_format = (AVPixelFormat)frame->format;
            last_serial = viddec.pkt_serial;
            last_vfilter_idx = vfilter_idx;
            frame_rate = av_buffersink_get_frame_rate(filt_out);
        }

        ret = av_buffersrc_add_frame(filt_in, frame);
        if (ret < 0)
            goto the_end;

        while (ret >= 0) {
            frame_last_returned_time = av_gettime_relative() / 1000000.0;

            ret = av_buffersink_get_frame_flags(filt_out, frame, 0);
            if (ret < 0) {
                if (ret == AVERROR_EOF)
                    viddec.finished = viddec.pkt_serial;
                ret = 0;
                break;
            }

            frame_last_filter_delay = av_gettime_relative() / 1000000.0 - frame_last_returned_time;
            if (fabs(frame_last_filter_delay) > AV_NOSYNC_THRESHOLD / 10.0)
                frame_last_filter_delay = 0;
            tb = av_buffersink_get_time_base(filt_out);
#endif
            duration = (frame_rate.num && frame_rate.den ? av_q2d(av_make_q(frame_rate.den, frame_rate.num)) : 0);
            pts = (frame->pts == AV_NOPTS_VALUE) ? NAN : frame->pts * av_q2d(tb);
            ret = queue_picture(frame, pts, duration, frame->pkt_pos, viddec.pkt_serial);
            av_frame_unref(frame);
#if CONFIG_AVFILTER
            if (videoq.serial != viddec.pkt_serial)
                break;
        }
#endif

        if (ret < 0)
            goto the_end;
    }
the_end:
#if CONFIG_AVFILTER
    avfilter_graph_free(&graph);
#endif
    av_frame_free(&frame);
    return 0;
}

static int subtitleThread(void* opaque)
{
    return static_cast<VideoState*>(opaque)->subtitle_thread();
}


int VideoState::subtitle_thread()
{
    //VideoState* is = (VideoState*)arg;
    Frame* sp;
    int got_subtitle;
    double pts;

    for (;;) {
        if (!(sp = subpq.peek_writable()))
            return 0;

        if ((got_subtitle = subdec.decode_frame(NULL, &sp->sub)) < 0)
            break;

        pts = 0;

        if (got_subtitle && sp->sub.format == 0) {
            if (sp->sub.pts != AV_NOPTS_VALUE)
                pts = sp->sub.pts / (double)AV_TIME_BASE;
            sp->pts = pts;
            sp->serial = subdec.pkt_serial;
            sp->width = subdec.avctx->width;
            sp->height = subdec.avctx->height;
            sp->uploaded = 0;

            // now we can update the picture count
            subpq.push();
        }
        else if (got_subtitle) {
            avsubtitle_free(&sp->sub);
        }
    }
    return 0;
}

static void sdlAudioCallback(void* opaque, Uint8* stream, int len)
{
    static_cast<VideoState*>(opaque)->sdl_audio_callback(stream, len);
}

void VideoState::sdl_audio_callback(Uint8* stream, int len)
{
    //VideoState* is = (VideoState*)opaque;
    int audio_size, len1;

    co->audio_callback_time = av_gettime_relative();

    while (len > 0) {
        if (audio_buf_index >= audio_buf_size) {
            audio_size = audio_decode_frame();
            if (audio_size < 0) {
                // if error, just output silence 
                audio_buf = NULL;
                audio_buf_size = SDL_AUDIO_MIN_BUFFER_SIZE / audio_tgt.frame_size * audio_tgt.frame_size;
            }
            else {
                if (show_mode != SHOW_MODE_VIDEO)
                    update_sample_display((int16_t*)audio_buf, audio_size);
                audio_buf_size = audio_size;
            }
            audio_buf_index = 0;
        }
        len1 = audio_buf_size - audio_buf_index;
        if (len1 > len)
            len1 = len;
        if (!muted && audio_buf && audio_volume == SDL_MIX_MAXVOLUME)
            memcpy(stream, (uint8_t*)audio_buf + audio_buf_index, len1);
        else {
            memset(stream, 0, len1);
            if (!muted && audio_buf)
                SDL_MixAudioFormat(stream, (uint8_t*)audio_buf + audio_buf_index, AUDIO_S16SYS, len1, audio_volume);
        }
        len -= len1;
        stream += len1;
        audio_buf_index += len1;
    }
    audio_write_buf_size = audio_buf_size - audio_buf_index;
    /* Let's assume the audio driver that is used by SDL has two periods. */
    if (!isnan(audio_clock)) {
        audclk.set_clock_at(audio_clock - (double)(2 * audio_hw_buf_size + audio_write_buf_size) / audio_tgt.bytes_per_sec, audio_clock_serial, co->audio_callback_time / 1000000.0);
        extclk.sync_clock_to_slave(&audclk);
    }
}

int VideoState::audio_open(int64_t wanted_channel_layout, int wanted_nb_channels, int wanted_sample_rate, struct AudioParams* audio_hw_params)
{
    SDL_AudioSpec wanted_spec, spec;
    const char* env;
    static const int next_nb_channels[] = { 0, 0, 1, 6, 2, 6, 4, 6 };
    static const int next_sample_rates[] = { 0, 44100, 48000, 96000, 192000 };
    int next_sample_rate_idx = FF_ARRAY_ELEMS(next_sample_rates) - 1;

    env = SDL_getenv("SDL_AUDIO_CHANNELS");
    if (env) {
        wanted_nb_channels = atoi(env);
        wanted_channel_layout = av_get_default_channel_layout(wanted_nb_channels);
    }
    if (!wanted_channel_layout || wanted_nb_channels != av_get_channel_layout_nb_channels(wanted_channel_layout)) {
        wanted_channel_layout = av_get_default_channel_layout(wanted_nb_channels);
        wanted_channel_layout &= ~AV_CH_LAYOUT_STEREO_DOWNMIX;
    }
    wanted_nb_channels = av_get_channel_layout_nb_channels(wanted_channel_layout);
    wanted_spec.channels = wanted_nb_channels;
    wanted_spec.freq = wanted_sample_rate;
    if (wanted_spec.freq <= 0 || wanted_spec.channels <= 0) {
        av_log(NULL, AV_LOG_ERROR, "Invalid sample rate or channel count!\n");
        return -1;
    }
    while (next_sample_rate_idx && next_sample_rates[next_sample_rate_idx] >= wanted_spec.freq)
        next_sample_rate_idx--;
    wanted_spec.format = AUDIO_S16SYS;
    wanted_spec.silence = 0;
    wanted_spec.samples = FFMAX(SDL_AUDIO_MIN_BUFFER_SIZE, 2 << av_log2(wanted_spec.freq / SDL_AUDIO_MAX_CALLBACKS_PER_SEC));
    //wanted_spec.callback = sdl_audio_callback;
    wanted_spec.callback = sdlAudioCallback;
    wanted_spec.userdata = this;
    while (!(disp->audio_dev = SDL_OpenAudioDevice(NULL, 0, &wanted_spec, &spec, SDL_AUDIO_ALLOW_FREQUENCY_CHANGE | SDL_AUDIO_ALLOW_CHANNELS_CHANGE))) {
        av_log(NULL, AV_LOG_WARNING, "SDL_OpenAudio (%d channels, %d Hz): %s\n",
            wanted_spec.channels, wanted_spec.freq, SDL_GetError());
        wanted_spec.channels = next_nb_channels[FFMIN(7, wanted_spec.channels)];
        if (!wanted_spec.channels) {
            wanted_spec.freq = next_sample_rates[next_sample_rate_idx--];
            wanted_spec.channels = wanted_nb_channels;
            if (!wanted_spec.freq) {
                av_log(NULL, AV_LOG_ERROR,
                    "No more combinations to try, audio open failed\n");
                return -1;
            }
        }
        wanted_channel_layout = av_get_default_channel_layout(wanted_spec.channels);
    }
    if (spec.format != AUDIO_S16SYS) {
        av_log(NULL, AV_LOG_ERROR,
            "SDL advised audio format %d is not supported!\n", spec.format);
        return -1;
    }
    if (spec.channels != wanted_spec.channels) {
        wanted_channel_layout = av_get_default_channel_layout(spec.channels);
        if (!wanted_channel_layout) {
            av_log(NULL, AV_LOG_ERROR,
                "SDL advised channel count %d is not supported!\n", spec.channels);
            return -1;
        }
    }

    audio_hw_params->fmt = AV_SAMPLE_FMT_S16;
    audio_hw_params->freq = spec.freq;
    audio_hw_params->channel_layout = wanted_channel_layout;
    audio_hw_params->channels = spec.channels;
    audio_hw_params->frame_size = av_samples_get_buffer_size(NULL, audio_hw_params->channels, 1, audio_hw_params->fmt, 1);
    audio_hw_params->bytes_per_sec = av_samples_get_buffer_size(NULL, audio_hw_params->channels, audio_hw_params->freq, audio_hw_params->fmt, 1);
    if (audio_hw_params->bytes_per_sec <= 0 || audio_hw_params->frame_size <= 0) {
        av_log(NULL, AV_LOG_ERROR, "av_samples_get_buffer_size failed\n");
        return -1;
    }
    return spec.size;
}

int VideoState::stream_component_open(int stream_index)
{
    //AVFormatContext* ic = ic;
    AVCodecContext* avctx;
    AVCodec* codec;
    const char* forced_codec_name = NULL;
    AVDictionary* opts = NULL;
    AVDictionaryEntry* t = NULL;
    int sample_rate, nb_channels;
    int64_t channel_layout;
    int ret = 0;
    int stream_lowres = co->lowres;

    if (stream_index < 0 || stream_index >= ic->nb_streams)
        return -1;

    avctx = avcodec_alloc_context3(NULL);
    if (!avctx)
        return AVERROR(ENOMEM);

    ret = avcodec_parameters_to_context(avctx, ic->streams[stream_index]->codecpar);
    if (ret < 0)
        goto fail;
    avctx->pkt_timebase = ic->streams[stream_index]->time_base;

    codec = avcodec_find_decoder(avctx->codec_id);

    switch (avctx->codec_type) {
    case AVMEDIA_TYPE_AUDIO: last_audio_stream = stream_index; forced_codec_name = co->audio_codec_name; break;
    case AVMEDIA_TYPE_SUBTITLE: last_subtitle_stream = stream_index; forced_codec_name = co->subtitle_codec_name; break;
    case AVMEDIA_TYPE_VIDEO: last_video_stream = stream_index; forced_codec_name = co->video_codec_name; break;
    }
    if (forced_codec_name)
        codec = avcodec_find_decoder_by_name(forced_codec_name);
    if (!codec) {
        if (forced_codec_name) 
            av_log(NULL, AV_LOG_WARNING, "No codec could be found with name '%s'\n", forced_codec_name);
        else                   
            av_log(NULL, AV_LOG_WARNING, "No decoder could be found for codec %s\n", avcodec_get_name(avctx->codec_id));
        ret = AVERROR(EINVAL);
        goto fail;
    }

    avctx->codec_id = codec->id;
    if (stream_lowres > codec->max_lowres) {
        av_log(avctx, AV_LOG_WARNING, "The maximum value for lowres supported by the decoder is %d\n",
            codec->max_lowres);
        stream_lowres = codec->max_lowres;
    }
    avctx->lowres = stream_lowres;

    if (co->fast)
        avctx->flags2 |= AV_CODEC_FLAG2_FAST;

    opts = filter_codec_opts(codec_opts, avctx->codec_id, ic, ic->streams[stream_index], codec);
    if (!av_dict_get(opts, "threads", NULL, 0))
        av_dict_set(&opts, "threads", "auto", 0);
    if (stream_lowres)
        av_dict_set_int(&opts, "lowres", stream_lowres, 0);
    if (avctx->codec_type == AVMEDIA_TYPE_VIDEO || avctx->codec_type == AVMEDIA_TYPE_AUDIO)
        av_dict_set(&opts, "refcounted_frames", "1", 0);
    if ((ret = avcodec_open2(avctx, codec, &opts)) < 0) {
        goto fail;
    }
    if ((t = av_dict_get(opts, "", NULL, AV_DICT_IGNORE_SUFFIX))) {
        av_log(NULL, AV_LOG_ERROR, "Option %s not found.\n", t->key);
        ret = AVERROR_OPTION_NOT_FOUND;
        goto fail;
    }

    eof = 0;
    ic->streams[stream_index]->discard = AVDISCARD_DEFAULT;
    switch (avctx->codec_type) {
    case AVMEDIA_TYPE_AUDIO:
#if CONFIG_AVFILTER
    {
        AVFilterContext* sink;

        audio_filter_src.freq = avctx->sample_rate;
        audio_filter_src.channels = avctx->channels;
        audio_filter_src.channel_layout = get_valid_channel_layout(avctx->channel_layout, avctx->channels);
        audio_filter_src.fmt = avctx->sample_fmt;
        if ((ret = configure_audio_filters(co->afilters, 0)) < 0)
            goto fail;
        sink = out_audio_filter;
        sample_rate = av_buffersink_get_sample_rate(sink);
        nb_channels = av_buffersink_get_channels(sink);
        channel_layout = av_buffersink_get_channel_layout(sink);
    }
#else
        sample_rate = avctx->sample_rate;
        nb_channels = avctx->channels;
        channel_layout = avctx->channel_layout;
#endif

        /* prepare audio output */
        if ((ret = audio_open(channel_layout, nb_channels, sample_rate, &audio_tgt)) < 0)
            goto fail;
        audio_hw_buf_size = ret;
        audio_src = audio_tgt;
        audio_buf_size = 0;
        audio_buf_index = 0;

        /* init averaging filter */
        audio_diff_avg_coef = exp(log(0.01) / AUDIO_DIFF_AVG_NB);
        audio_diff_avg_count = 0;
        /* since we do not have a precise anough audio FIFO fullness,
           we correct audio sync only if larger than this threshold */
        audio_diff_threshold = (double)(audio_hw_buf_size) / audio_tgt.bytes_per_sec;

        audio_stream = stream_index;
        audio_st = ic->streams[stream_index];

        auddec.init(avctx, &audioq, continue_read_thread, flush_pkt);
        if ((ic->iformat->flags & (AVFMT_NOBINSEARCH | AVFMT_NOGENSEARCH | AVFMT_NO_BYTE_SEEK)) && !ic->iformat->read_seek) {
            auddec.start_pts = audio_st->start_time;
            auddec.start_pts_tb = audio_st->time_base;
        }
        if ((ret = auddec.start(audioThread, "audio_decoder", this)) < 0)
            goto out;
        SDL_PauseAudioDevice(disp->audio_dev, 0);
        break;
    case AVMEDIA_TYPE_VIDEO:
        video_stream = stream_index;
        video_st = ic->streams[stream_index];

        viddec.init(avctx, &videoq, continue_read_thread, flush_pkt);
        if ((ret = viddec.start(videoThread, "video_decoder", this)) < 0)
            goto out;
        queue_attachments_req = 1;
        break;
    case AVMEDIA_TYPE_SUBTITLE:
        subtitle_stream = stream_index;
        subtitle_st = ic->streams[stream_index];

        subdec.init(avctx, &subtitleq, continue_read_thread, flush_pkt);
        if ((ret = subdec.start(subtitleThread, "subtitle_decoder", this)) < 0)
            goto out;
        break;
    default:
        break;
    }
    goto out;

fail:
    avcodec_free_context(&avctx);
out:
    av_dict_free(&opts);

    return ret;
}

int VideoState::stream_has_enough_packets(AVStream* st, int stream_id, PacketQueue* queue) {
    return stream_id < 0 ||
        queue->abort_request ||
        (st->disposition & AV_DISPOSITION_ATTACHED_PIC) ||
        queue->nb_packets > MIN_FRAMES && (!queue->duration || av_q2d(st->time_base) * queue->duration > 1.0);
}

int VideoState::is_realtime(AVFormatContext* s)
{
    if (!strcmp(s->iformat->name, "rtp")
        || !strcmp(s->iformat->name, "rtsp")
        || !strcmp(s->iformat->name, "sdp")
        )
        return 1;

    if (s->pb && (!strncmp(s->url, "rtp:", 4)
        || !strncmp(s->url, "udp:", 4)
        )
        )
        return 1;
    return 0;
}

static int decode_interrupt_cb(void* ctx)
{
    VideoState* is = (VideoState*)ctx;
    return is->abort_request;
}

static int readThread(void* opaque)
{
    return static_cast<VideoState*>(opaque)->read_thread();
}

int VideoState::read_thread()
{
    //VideoState* is = (VideoState*)arg;
    AVFormatContext* ictx = NULL;
    int err, i, ret;
    int st_index[AVMEDIA_TYPE_NB];
    AVPacket pkt1, * pkt = &pkt1;
    int64_t stream_start_time;
    int pkt_in_play_range = 0;
    AVDictionaryEntry* t;
    SDL_mutex* wait_mutex = SDL_CreateMutex();
    int scan_all_pmts_set = 0;
    int64_t pkt_ts;

    if (!wait_mutex) {
        av_log(NULL, AV_LOG_FATAL, "SDL_CreateMutex(): %s\n", SDL_GetError());
        ret = AVERROR(ENOMEM);
        goto fail;
    }

    memset(st_index, -1, sizeof(st_index));
    last_video_stream = video_stream = -1;
    last_audio_stream = audio_stream = -1;
    last_subtitle_stream = subtitle_stream = -1;
    eof = 0;

    ictx = avformat_alloc_context();
    if (!ictx) {
        av_log(NULL, AV_LOG_FATAL, "Could not allocate context.\n");
        ret = AVERROR(ENOMEM);
        goto fail;
    }
    ictx->interrupt_callback.callback = decode_interrupt_cb;
    ictx->interrupt_callback.opaque = this;
    if (!av_dict_get(format_opts, "scan_all_pmts", NULL, AV_DICT_MATCH_CASE)) {
        av_dict_set(&format_opts, "scan_all_pmts", "1", AV_DICT_DONT_OVERWRITE);
        scan_all_pmts_set = 1;
    }
    //err = avformat_open_input(&ic, is->filename, is->iformat, &format_opts);

    fprintf(stderr, "is->filename: %s\n", filename);

    err = avformat_open_input(&ictx, filename, NULL, NULL);
    if (err < 0) {
        print_error(filename, err);
        ret = -1;
        goto fail;
    }
    if (scan_all_pmts_set)
        av_dict_set(&format_opts, "scan_all_pmts", NULL, AV_DICT_MATCH_CASE);

    if ((t = av_dict_get(format_opts, "", NULL, AV_DICT_IGNORE_SUFFIX))) {
        av_log(NULL, AV_LOG_ERROR, "Option %s not found.\n", t->key);
        ret = AVERROR_OPTION_NOT_FOUND;
        goto fail;
    }
    ic = ictx;

    if (co->genpts)
        ictx->flags |= AVFMT_FLAG_GENPTS;

    //av_format_inject_global_side_data(ic);

    if (co->find_stream_info) {
        AVDictionary** opts = setup_find_stream_info_opts(ictx, codec_opts);
        int orig_nb_streams = ictx->nb_streams;

        err = avformat_find_stream_info(ictx, opts);

        for (i = 0; i < orig_nb_streams; i++)
            av_dict_free(&opts[i]);
        av_freep(&opts);

        if (err < 0) {
            av_log(NULL, AV_LOG_WARNING,
                "%s: could not find codec parameters\n", filename);
            ret = -1;
            goto fail;
        }
    }

    if (ictx->pb)
        ictx->pb->eof_reached = 0; // FIXME hack, ffplay maybe should not use avio_feof() to test for the end

    if (co->seek_by_bytes < 0)
        co->seek_by_bytes = !!(ictx->iformat->flags & AVFMT_TS_DISCONT) && strcmp("ogg", ictx->iformat->name);

    max_frame_duration = (ictx->iformat->flags & AVFMT_TS_DISCONT) ? 10.0 : 3600.0;

    if (!co->window_title && (t = av_dict_get(ictx->metadata, "title", NULL, 0)))
        co->window_title = av_asprintf("%s - %s", t->value, co->input_filename);

    /* if seeking requested, we execute it */
    if (co->start_time != AV_NOPTS_VALUE) {
        int64_t timestamp;

        timestamp = co->start_time;
        /* add the stream start time */
        if (ictx->start_time != AV_NOPTS_VALUE)
            timestamp += ictx->start_time;
        ret = avformat_seek_file(ictx, -1, INT64_MIN, timestamp, INT64_MAX, 0);
        if (ret < 0) {
            av_log(NULL, AV_LOG_WARNING, "%s: could not seek to position %0.3f\n",
                filename, (double)timestamp / AV_TIME_BASE);
        }
    }

    realtime = is_realtime(ictx);

    //if (show_status)
    av_dump_format(ictx, 0, filename, 0);

    for (i = 0; i < ictx->nb_streams; i++) {
        AVStream* st = ictx->streams[i];
        enum AVMediaType type = st->codecpar->codec_type;
        st->discard = AVDISCARD_ALL;
        if (type >= 0 && co->wanted_stream_spec[type] && st_index[type] == -1)
            if (avformat_match_stream_specifier(ictx, st, co->wanted_stream_spec[type]) > 0)
                st_index[type] = i;
    }
    for (i = 0; i < AVMEDIA_TYPE_NB; i++) {
        if (co->wanted_stream_spec[i] && st_index[i] == -1) {
            av_log(NULL, AV_LOG_ERROR, "Stream specifier %s does not match any %s stream\n", co->wanted_stream_spec[i], av_get_media_type_string((AVMediaType)i));
            st_index[i] = INT_MAX;
        }
    }

    if (!co->video_disable)
        st_index[AVMEDIA_TYPE_VIDEO] =
        av_find_best_stream(ictx, AVMEDIA_TYPE_VIDEO,
            st_index[AVMEDIA_TYPE_VIDEO], -1, NULL, 0);
    if (!co->audio_disable)
        st_index[AVMEDIA_TYPE_AUDIO] =
        av_find_best_stream(ictx, AVMEDIA_TYPE_AUDIO,
            st_index[AVMEDIA_TYPE_AUDIO],
            st_index[AVMEDIA_TYPE_VIDEO],
            NULL, 0);
    if (!co->video_disable && !co->subtitle_disable)
        st_index[AVMEDIA_TYPE_SUBTITLE] =
        av_find_best_stream(ictx, AVMEDIA_TYPE_SUBTITLE,
            st_index[AVMEDIA_TYPE_SUBTITLE],
            (st_index[AVMEDIA_TYPE_AUDIO] >= 0 ?
                st_index[AVMEDIA_TYPE_AUDIO] :
                st_index[AVMEDIA_TYPE_VIDEO]),
            NULL, 0);

    show_mode = co->show_mode;
    if (st_index[AVMEDIA_TYPE_VIDEO] >= 0) {
        AVStream* st = ictx->streams[st_index[AVMEDIA_TYPE_VIDEO]];
        AVCodecParameters* codecpar = st->codecpar;
        fprintf(stderr, "read codecpar extradata size %d\n\n", codecpar->extradata_size);
        AVRational sar = av_guess_sample_aspect_ratio(ictx, st, NULL);
        if (codecpar->width)
            set_default_window_size(codecpar->width, codecpar->height, sar);
    }

    /* open the streams */
    if (st_index[AVMEDIA_TYPE_AUDIO] >= 0) {
        stream_component_open(st_index[AVMEDIA_TYPE_AUDIO]);
    }

    ret = -1;
    if (st_index[AVMEDIA_TYPE_VIDEO] >= 0) {
        ret = stream_component_open(st_index[AVMEDIA_TYPE_VIDEO]);
    }
    if (show_mode == SHOW_MODE_NONE)
        show_mode = ret >= 0 ? SHOW_MODE_VIDEO : SHOW_MODE_RDFT;

    if (st_index[AVMEDIA_TYPE_SUBTITLE] >= 0) {
        stream_component_open(st_index[AVMEDIA_TYPE_SUBTITLE]);
    }

    if (video_stream < 0 && audio_stream < 0) {
        av_log(NULL, AV_LOG_FATAL, "Failed to open file '%s' or configure filtergraph\n",
            filename);
        ret = -1;
        goto fail;
    }

    if (co->infinite_buffer < 0 && realtime)
        co->infinite_buffer = 1;

    for (;;) {
        if (abort_request)
            break;
        if (paused != last_paused) {
            last_paused = paused;
            if (paused)
                read_pause_return = av_read_pause(ictx);
            else
                av_read_play(ictx);
        }
#if CONFIG_RTSP_DEMUXER || CONFIG_MMSH_PROTOCOL
        if (paused &&
            (!strcmp(ictx->iformat->name, "rtsp") ||
                (ictx->pb && !strncmp(co->input_filename, "mmsh:", 5)))) {
            /* wait 10 ms to avoid trying to get another packet */
            /* XXX: horrible */
            SDL_Delay(10);
            continue;
        }
#endif
        if (seek_req) {
            int64_t seek_target = seek_pos;
            int64_t seek_min = seek_rel > 0 ? seek_target - seek_rel + 2 : INT64_MIN;
            int64_t seek_max = seek_rel < 0 ? seek_target - seek_rel - 2 : INT64_MAX;
            // FIXME the +-2 is due to rounding being not done in the correct direction in generation
            //      of the seek_pos/seek_rel variables

            ret = avformat_seek_file(ic, -1, seek_min, seek_target, seek_max, seek_flags);
            if (ret < 0) {
                av_log(NULL, AV_LOG_ERROR,
                    "%s: error while seeking\n", ic->url);
            }
            else {
                if (audio_stream >= 0) {
                    audioq.flush();
                    audioq.put(flush_pkt);
                }
                if (subtitle_stream >= 0) {
                    subtitleq.flush();
                    subtitleq.put(flush_pkt);
                }
                if (video_stream >= 0) {
                    videoq.flush();
                    videoq.put(flush_pkt);
                }
                if (seek_flags & AVSEEK_FLAG_BYTE) {
                    extclk.set_clock(NAN, 0);
                }
                else {
                    extclk.set_clock(seek_target / (double)AV_TIME_BASE, 0);
                }
            }
            seek_req = 0;
            queue_attachments_req = 1;
            eof = 0;
            if (paused)
                step_to_next_frame();
        }
        if (queue_attachments_req) {
            if (video_st && video_st->disposition & AV_DISPOSITION_ATTACHED_PIC) {
                AVPacket copy = { 0 };
                if ((ret = av_packet_ref(&copy, &video_st->attached_pic)) < 0)
                    goto fail;
                videoq.put(&copy);
                videoq.put_nullpacket(video_stream);
            }
            queue_attachments_req = 0;
        }

        /* if the queue are full, no need to read more */
        if (co->infinite_buffer < 1 &&
            (audioq.size + videoq.size + subtitleq.size > MAX_QUEUE_SIZE
                || (stream_has_enough_packets(audio_st, audio_stream, &audioq) &&
                    stream_has_enough_packets(video_st, video_stream, &videoq) &&
                    stream_has_enough_packets(subtitle_st, subtitle_stream, &subtitleq)))) {
            /* wait 10 ms */
            SDL_LockMutex(wait_mutex);
            SDL_CondWaitTimeout(continue_read_thread, wait_mutex, 10);
            SDL_UnlockMutex(wait_mutex);
            continue;
        }
        if (!paused &&
            (!audio_st || (auddec.finished == audioq.serial && sampq.nb_remaining() == 0)) &&
            (!video_st || (viddec.finished == videoq.serial && pictq.nb_remaining() == 0))) {
            if (co->loop != 1 && (!co->loop || --co->loop)) {
                stream_seek(co->start_time != AV_NOPTS_VALUE ? co->start_time : 0, 0, 0);
            }
            else if (co->autoexit) {
                ret = AVERROR_EOF;
                goto fail;
            }
        }
        ret = av_read_frame(ictx, pkt);
        if (ret < 0) {
            if ((ret == AVERROR_EOF || avio_feof(ictx->pb)) && !eof) {
                if (video_stream >= 0)
                    videoq.put_nullpacket(video_stream);
                if (audio_stream >= 0)
                    audioq.put_nullpacket(audio_stream);
                if (subtitle_stream >= 0)
                    subtitleq.put_nullpacket(subtitle_stream);
                eof = 1;
            }
            if (ictx->pb && ictx->pb->error)
                break;
            SDL_LockMutex(wait_mutex);
            SDL_CondWaitTimeout(continue_read_thread, wait_mutex, 10);
            SDL_UnlockMutex(wait_mutex);
            continue;
        }
        else {
            eof = 0;
        }
        /* check if packet is in play range specified by user, then queue, otherwise discard */
        stream_start_time = ictx->streams[pkt->stream_index]->start_time;
        pkt_ts = pkt->pts == AV_NOPTS_VALUE ? pkt->dts : pkt->pts;
        pkt_in_play_range = co->duration == AV_NOPTS_VALUE ||
            (pkt_ts - (stream_start_time != AV_NOPTS_VALUE ? stream_start_time : 0)) *
            av_q2d(ictx->streams[pkt->stream_index]->time_base) -
            (double)(co->start_time != AV_NOPTS_VALUE ? co->start_time : 0) / 1000000
            <= ((double)co->duration / 1000000);
        if (pkt->stream_index == audio_stream && pkt_in_play_range) {
            audioq.put(pkt);
        }
        else if (pkt->stream_index == video_stream && pkt_in_play_range
            && !(video_st->disposition & AV_DISPOSITION_ATTACHED_PIC)) {
            videoq.put(pkt);
        }
        else if (pkt->stream_index == subtitle_stream && pkt_in_play_range) {
            subtitleq.put(pkt);
        }
        else {
            av_packet_unref(pkt);
        }
    }

    ret = 0;
fail:
    if (ictx && !ic)
        avformat_close_input(&ictx);

    if (ret != 0) {
        SDL_Event event;

        event.type = FF_QUIT_EVENT;
        event.user.data1 = this;
        SDL_PushEvent(&event);
    }
    SDL_DestroyMutex(wait_mutex);
    return 0;
}

VideoState* VideoState::stream_open(QMainWindow *mw, const char* filename, AVInputFormat* iformat, CommandOptions* co, Display* disp)
{
    VideoState* is;

    is = (VideoState*)av_mallocz(sizeof(VideoState));

    //is = new VideoState();

    is->co = co;
    is->disp = disp;

    if (!is)
        return NULL;

    is->filename = av_strdup(((MainWindow*)mw)->filename.toLatin1().data());
    //is->filename = av_strdup(filename);
    //if (!is->filename)
    //    goto fail;

    mw->setWindowTitle(((MainWindow*)mw)->filename);
    cout << "VideoState::stream_open: mw->filename: " << ((MainWindow*)mw)->filename.toStdString() << endl;
    //is->filename = ((MainWindow*)mw)->filename.toLatin1().data();

    is->iformat = iformat;
    is->ytop = 0;
    is->xleft = 0;

    if (is->pictq.init(&is->videoq, VIDEO_PICTURE_QUEUE_SIZE, 1) < 0)
        goto fail;
    if (is->subpq.init(&is->subtitleq, SUBPICTURE_QUEUE_SIZE, 0) < 0)
        goto fail;
    if (is->sampq.init(&is->audioq, SAMPLE_QUEUE_SIZE, 1) < 0)
        goto fail;

    if (is->videoq.init() < 0)
        goto fail;

    if (is->audioq.init() < 0)
        goto fail;

    if (is->subtitleq.init() < 0)
        goto fail;

    if (!(is->continue_read_thread = SDL_CreateCond())) {
        av_log(NULL, AV_LOG_FATAL, "SDL_CreateCond(): %s\n", SDL_GetError());
        goto fail;
    }

    is->vidclk.init_clock(&is->videoq.serial);
    is->audclk.init_clock(&is->audioq.serial);
    is->extclk.init_clock(&is->extclk.serial);
    is->audio_clock_serial = -1;

    if (co->startup_volume < 0)
        av_log(NULL, AV_LOG_WARNING, "-volume=%d < 0, setting to 0\n", co->startup_volume);
    if (co->startup_volume > 100)
        av_log(NULL, AV_LOG_WARNING, "-volume=%d > 100, setting to 100\n", co->startup_volume);
    co->startup_volume = av_clip(co->startup_volume, 0, 100);
    co->startup_volume = av_clip(SDL_MIX_MAXVOLUME * co->startup_volume / 100, 0, SDL_MIX_MAXVOLUME);
    is->audio_volume = co->startup_volume;
    is->muted = 0;
    is->av_sync_type = co->av_sync_type;
    is->read_tid = SDL_CreateThread(readThread, "read_thread", is);

    if (!is->read_tid) {
        av_log(NULL, AV_LOG_FATAL, "SDL_CreateThread(): %s\n", SDL_GetError());
    fail:
        //stream_close(is);
        is->stream_close();
        return NULL;
    }

    return is;
}

void VideoState::refresh_loop_wait_event(SDL_Event* event) {
    double remaining_time = 0.0;
    SDL_PumpEvents();
    while (!SDL_PeepEvents(event, 1, SDL_GETEVENT, SDL_FIRSTEVENT, SDL_LASTEVENT)) {
        if (!co->cursor_hidden && av_gettime_relative() - co->cursor_last_shown > CURSOR_HIDE_DELAY) {
            SDL_ShowCursor(0);
            co->cursor_hidden = 1;
        }
        if (remaining_time > 0.0)
            av_usleep((int64_t)(remaining_time * 1000000.0));
        remaining_time = REFRESH_RATE;
        if (show_mode != SHOW_MODE_NONE && (!paused || force_refresh))
            video_refresh(&remaining_time);
        SDL_PumpEvents();
    }
}

/*
void VideoState::event_loop()
{
    SDL_Event event;
    double incr, pos, frac;

    for (;;) {
        double x;
        refresh_loop_wait_event(&event);
        switch (event.type) {
        case SDL_KEYDOWN:
            if (co->exit_on_keydown || event.key.keysym.sym == SDLK_ESCAPE || event.key.keysym.sym == SDLK_q) {
                do_exit();
                break;
            }
            break;
        case SDL_QUIT:
        case FF_QUIT_EVENT:
            do_exit();
            break;
        default:
            break;
        }
    }
}
*/

void VideoState::do_exit()
{
    stream_close();

    if (disp->renderer)
        SDL_DestroyRenderer(disp->renderer);
    if (disp->window)
        SDL_DestroyWindow(disp->window);
    uninit_opts();
#if CONFIG_AVFILTER
    av_freep(&co->vfilters_list);
#endif
    avformat_network_deinit();
    if (co->show_status)
        printf("\n");
    SDL_Quit();
    av_log(NULL, AV_LOG_QUIET, "%s", "");
    exit(0);
}

