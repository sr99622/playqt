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

    while (looping) {
        feed();
    }

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

    if (event.type == FF_QUIT_EVENT) {
        looping = false;
    }
}
