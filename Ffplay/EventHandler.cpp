#include "EventHandler.h"
#include "mainwindow.h"

EventHandler::EventHandler(QMainWindow *parent)
{
    mainWindow = parent;
}

void EventHandler::event_loop()
{
    running = true;
    elapsed = 0;
    total = 0;
    percentage = 0;

    while (running) {
        feed();
    }

    if (MW->is != nullptr)
        MW->is->stream_close();
    MW->is = nullptr;
}

void EventHandler::guiUpdate(int flag)
{
    if (MW->is->paused) {
        //cout << "EventHandler::guiUpdate: " << flag << endl;
    }
    else {
        double remaining_time = REFRESH_RATE;
        if (MW->is->show_mode != SHOW_MODE_NONE && (!MW->is->paused || MW->is->force_refresh))
            MW->is->video_refresh(&remaining_time);
    }
}

void EventHandler::feed()
{
    if (!MW->is)
        return;

    //cout << QTime::currentTime().toString("hh:mm:ss.zzz").toStdString() << endl;
    MW->is->refresh_loop_wait_event(&event);
    DisplayContainer *dc = MW->mainPanel->displayContainer;

    if (event.type == MW->sdlCustomEventType) {
        switch (event.user.code) {
        case FILE_POSITION_UPDATE:
            elapsed = *(double*)(event.user.data1);
            total = *(double*)(event.user.data2);
            percentage = (1000 * elapsed) / total;
            dc->slider->setValue(percentage);
            dc->elapsed->setText(MW->is->formatTime(elapsed));
            dc->total->setText(MW->is->formatTime(total));
            break;
        case SLIDER_POSITION_UPDATE:
            percentage = *(int*)(event.user.data1);
            ts = percentage * MW->is->ic->duration / 1000;
            if (MW->is->ic->start_time != AV_NOPTS_VALUE)
                ts += MW->is->ic->start_time;
            MW->is->stream_seek(ts, 0, 0);
            break;
        case PAUSE:
            MW->is->toggle_pause();
            cout << "PAUSE" << endl;
            break;
        case REWIND:
            MW->is->rewind();
            cout << "REWIND" << endl;
            break;
        case FASTFORWARD:
            MW->is->fastforward();
            cout << "FASTFORWARD" << endl;
            break;
        case GUI_UPDATE:
            //cout << "GUI_UPDATE" << endl;
            //MW->is->video_image_display();
            //MW->is->video_display();
            remaining_time = REFRESH_RATE;
            if (MW->is->show_mode != SHOW_MODE_NONE && (!MW->is->paused || MW->is->force_refresh))
                MW->is->video_refresh(&remaining_time);

            break;
        }
    }
    else if (event.type == REWIND) {

    }
    else if (event.type == FASTFORWARD) {

    }
    else if (event.type == FF_QUIT_EVENT) {
        running = false;
    }
}
