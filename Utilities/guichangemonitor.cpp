#include "guichangemonitor.h"

GuiChangeMonitor::GuiChangeMonitor()
{
    setAutoDelete(false);
}

void GuiChangeMonitor::setCountdown(int arg)
{
    mutex.lock();
    countdown = arg;
    mutex.unlock();
}

void GuiChangeMonitor::run()
{
    running = true;
    countdown = 10;
    while (countdown > 0) {
        QThread::msleep(timeout);
        countdown--;
    }
    running = false;

    emit done();
}
