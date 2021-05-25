#include "EventHandler.h"

void EventHandler::event_loop(VideoState* cur_stream)
{
    SDL_Event event;
    double incr, pos, frac;

    for (;;) {
        double x;
        cur_stream->refresh_loop_wait_event(&event);
        switch (event.type) {
        case SDL_KEYDOWN:
            if (cur_stream->co->exit_on_keydown || event.key.keysym.sym == SDLK_ESCAPE || event.key.keysym.sym == SDLK_q) {
                //do_exit(cur_stream);
                goto exit;
                break;
            }
            // If we don't yet have a window, skip all key events, because read_thread might still be initializing...
            if (!cur_stream->width)
                continue;
            switch (event.key.keysym.sym) {
            case SDLK_f:
                cur_stream->toggle_full_screen();
                cur_stream->force_refresh = 1;
                break;
            case SDLK_p:
            case SDLK_SPACE:
                cur_stream->toggle_pause();
                break;
            case SDLK_m:
                cur_stream->toggle_mute();
                break;
            case SDLK_KP_MULTIPLY:
            case SDLK_0:
                cur_stream->update_volume(1, SDL_VOLUME_STEP);
                break;
            case SDLK_KP_DIVIDE:
            case SDLK_9:
                cur_stream->update_volume(-1, SDL_VOLUME_STEP);
                break;
            case SDLK_s: // S: Step to next frame
                cur_stream->step_to_next_frame();
                break;
            case SDLK_a:
                cur_stream->stream_cycle_channel(AVMEDIA_TYPE_AUDIO);
                break;
            case SDLK_v:
                cur_stream->stream_cycle_channel(AVMEDIA_TYPE_VIDEO);
                break;
            case SDLK_c:
                cur_stream->stream_cycle_channel(AVMEDIA_TYPE_VIDEO);
                cur_stream->stream_cycle_channel(AVMEDIA_TYPE_AUDIO);
                cur_stream->stream_cycle_channel(AVMEDIA_TYPE_SUBTITLE);
                break;
            case SDLK_t:
                cur_stream->stream_cycle_channel(AVMEDIA_TYPE_SUBTITLE);
                break;
            case SDLK_w:
#if CONFIG_AVFILTER
                if (cur_stream->show_mode == SHOW_MODE_VIDEO && cur_stream->vfilter_idx < cur_stream->co->nb_vfilters - 1) {
                    if (++cur_stream->vfilter_idx >= cur_stream->co->nb_vfilters)
                        cur_stream->vfilter_idx = 0;
                }
                else {
                    cur_stream->vfilter_idx = 0;
                    cur_stream->toggle_audio_display();
                }
#else
                cur_stream->toggle_audio_display();
#endif
                break;
            case SDLK_PAGEUP:
                if (cur_stream->ic->nb_chapters <= 1) {
                    incr = 600.0;
                    goto do_seek;
                }
                cur_stream->seek_chapter(1);
                break;
            case SDLK_PAGEDOWN:
                if (cur_stream->ic->nb_chapters <= 1) {
                    incr = -600.0;
                    goto do_seek;
                }
                cur_stream->seek_chapter(-1);
                break;
            case SDLK_LEFT:
                incr = cur_stream->co->seek_interval ? -cur_stream->co->seek_interval : -10.0;
                goto do_seek;
            case SDLK_RIGHT:
                incr = cur_stream->co->seek_interval ? cur_stream->co->seek_interval : 10.0;
                goto do_seek;
            case SDLK_UP:
                incr = 60.0;
                goto do_seek;
            case SDLK_DOWN:
                incr = -60.0;
            do_seek:
                if (cur_stream->co->seek_by_bytes) {
                    pos = -1;
                    if (pos < 0 && cur_stream->video_stream >= 0)
                        pos = cur_stream->pictq.last_pos();
                    if (pos < 0 && cur_stream->audio_stream >= 0)
                        pos = cur_stream->sampq.last_pos();
                    if (pos < 0)
                        pos = avio_tell(cur_stream->ic->pb);
                    if (cur_stream->ic->bit_rate)
                        incr *= cur_stream->ic->bit_rate / 8.0;
                    else
                        incr *= 180000.0;
                    pos += incr;
                    cur_stream->stream_seek(pos, incr, 1);
                }
                else {
                    pos = cur_stream->get_master_clock();
                    if (isnan(pos))
                        pos = (double)cur_stream->seek_pos / AV_TIME_BASE;
                    pos += incr;
                    if (cur_stream->ic->start_time != AV_NOPTS_VALUE && pos < cur_stream->ic->start_time / (double)AV_TIME_BASE)
                        pos = cur_stream->ic->start_time / (double)AV_TIME_BASE;
                    cur_stream->stream_seek((int64_t)(pos * AV_TIME_BASE), (int64_t)(incr * AV_TIME_BASE), 0);
                }
                break;
            default:
                break;
            }
            break;
        case SDL_MOUSEBUTTONDOWN:
            if (cur_stream->co->exit_on_mousedown) {
                //do_exit(cur_stream);
                goto exit;
                break;
            }
            if (event.button.button == SDL_BUTTON_LEFT) {
                static int64_t last_mouse_left_click = 0;
                if (av_gettime_relative() - last_mouse_left_click <= 500000) {
                    cur_stream->toggle_full_screen();
                    cur_stream->force_refresh = 1;
                    last_mouse_left_click = 0;
                }
                else {
                    last_mouse_left_click = av_gettime_relative();
                }
            }
        case SDL_MOUSEMOTION:
            if (cur_stream->co->cursor_hidden) {
                SDL_ShowCursor(1);
                cur_stream->co->cursor_hidden = 0;
            }
            cur_stream->co->cursor_last_shown = av_gettime_relative();
            if (event.type == SDL_MOUSEBUTTONDOWN) {
                if (event.button.button != SDL_BUTTON_RIGHT)
                    break;
                x = event.button.x;
            }
            else {
                if (!(event.motion.state & SDL_BUTTON_RMASK))
                    break;
                x = event.motion.x;
            }
            if (cur_stream->co->seek_by_bytes || cur_stream->ic->duration <= 0) {
                uint64_t size = avio_size(cur_stream->ic->pb);
                cur_stream->stream_seek(size * x / cur_stream->width, 0, 1);
            }
            else {
                int64_t ts;
                int ns, hh, mm, ss;
                int tns, thh, tmm, tss;
                tns = cur_stream->ic->duration / 1000000LL;
                thh = tns / 3600;
                tmm = (tns % 3600) / 60;
                tss = (tns % 60);
                frac = x / cur_stream->width;
                ns = frac * tns;
                hh = ns / 3600;
                mm = (ns % 3600) / 60;
                ss = (ns % 60);
                av_log(NULL, AV_LOG_INFO,
                    "Seek to %2.0f%% (%2d:%02d:%02d) of total duration (%2d:%02d:%02d)       \n", frac * 100,
                    hh, mm, ss, thh, tmm, tss);
                ts = frac * cur_stream->ic->duration;
                if (cur_stream->ic->start_time != AV_NOPTS_VALUE)
                    ts += cur_stream->ic->start_time;
                cur_stream->stream_seek(ts, 0, 0);
            }
            break;
        case SDL_WINDOWEVENT:
            switch (event.window.event) {
            case SDL_WINDOWEVENT_RESIZED:
                cur_stream->co->screen_width = cur_stream->width = event.window.data1;
                cur_stream->co->screen_height = cur_stream->height = event.window.data2;
                if (cur_stream->vis_texture) {
                    SDL_DestroyTexture(cur_stream->vis_texture);
                    cur_stream->vis_texture = NULL;
                }
            case SDL_WINDOWEVENT_EXPOSED:
                cur_stream->force_refresh = 1;
            }
            break;
        case SDL_QUIT:
        case FF_QUIT_EVENT:
            //do_exit(cur_stream);
            goto exit;
            break;
        default:
            break;
        }
    }
exit:
    return;
}

