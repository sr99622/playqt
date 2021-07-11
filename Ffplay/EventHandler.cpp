#include "EventHandler.h"
#include "mainwindow.h"

EventHandler::EventHandler(QMainWindow *parent)
{
    mainWindow = parent;
}

void EventHandler::event_loop()
{
    looping = true;
    running = true;
    elapsed = 0;
    total = 0;
    percentage = 0;

    MW->is->refresh_loop_flush_event(&event);

    while (looping) {
        feed();
    }

    if (MW->is != nullptr)
        MW->is->stream_close();

    MW->is = nullptr;

    DisplayContainer *dc = MW->mainPanel->displayContainer;
    dc->slider->setValue(0);
    dc->elapsed->setText("");
    dc->total->setText("");
    MW->repaint();
    running = false;

}

void EventHandler::feed()
{
    //cout << "event handler: " << QTime::currentTime().toString("hh:mm:ss.zzz").toStdString() << endl;

    if (event.type == MW->sdlCustomEventType && event.user.code == FLUSH)
        MW->is->refresh_loop_flush_event(&event);
    else
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
    else if (event.type == FF_QUIT_EVENT) {
        cout << "QUIT EVENT" << endl;
        looping = false;
        /*
        dc->slider->setValue(0);
        dc->elapsed->setText("");
        dc->total->setText("");
        MW->repaint();
        */
    }
}
