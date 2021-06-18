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

void EventHandler::feed()
{
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
        }
    }
    else if (event.type == SDL_KEYDOWN) {
        if (event.key.keysym.sym == SDLK_ESCAPE) {
            running = false;
        }
    }
}
