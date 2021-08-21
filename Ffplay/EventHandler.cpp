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

    while (looping) {
        feed();
    }

    MW->control()->engageFilter->setChecked(false);
    MW->filter()->engageFilter->setChecked(false);

    if (MW->is != nullptr)
        MW->is->stream_close();

    MW->is = nullptr;

    MW->dc()->slider->setValue(0);
    MW->dc()->elapsed->setText("");
    MW->dc()->total->setText("");
    MW->control()->btnPlay->setStyleSheet(MW->control()->getButtonStyle("play"));
    MW->repaint();
    running = false;

}

void EventHandler::feed()
{
    MW->is->refresh_loop_wait_event(&event);

    if (event.type == MW->sdlCustomEventType) {
        if (event.user.code == FILE_POSITION_UPDATE) {
            elapsed = *(double*)(event.user.data1);
            total = *(double*)(event.user.data2);
            percentage = (1000 * elapsed) / total;
            MW->dc()->slider->setValue(percentage);
            MW->dc()->elapsed->setText(MW->is->formatTime(elapsed));
            MW->dc()->total->setText(MW->is->formatTime(total));
        }
    }
    else if (event.type == FF_QUIT_EVENT) {
        looping = false;
    }
}
