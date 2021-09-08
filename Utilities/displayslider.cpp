#include "displayslider.h"
#include "mainwindow.h"

DisplaySlider::DisplaySlider(QMainWindow *parent) : QSlider(parent)
{
    mainWindow = parent;
    setMouseTracking(true);
}

bool DisplaySlider::event(QEvent *e)
{
    if (e->type() == QEvent::Leave)
        QToolTip::hideText();

    return QSlider::event(e);
}

void DisplaySlider::mouseMoveEvent(QMouseEvent *e)
{

    if (last_position_x != e->position().x())
        QToolTip::hideText();

    if (MW->is) {
        double percentage = e->position().x() / (double)width();
        double position = percentage * MW->is->total;
        QString output = MW->is->formatTime(position);
        if (output.startsWith("00:"))
            output = output.mid(3);

        const QPoint pos = mapToGlobal(QPoint(e->position().x(), geometry().top() - 40));
        QToolTip::showText(pos, output);
        last_position_x = e->position().x();
    }
}

void DisplaySlider::mousePressEvent(QMouseEvent *e)
{
    //QSlider::mousePressEvent(e);
}

void DisplaySlider::mouseReleaseEvent(QMouseEvent *e)
{
    if (!MW->is)
        return;

    MW->filterChain->disengaged = true;

    tick = 1000 * e->position().x() / width();

    double ts = tick * MW->is->ic->duration / 1000;
    if (MW->is->ic->start_time != AV_NOPTS_VALUE)
        ts += MW->is->ic->start_time;
    MW->is->stream_seek(ts, 0, 0);

    MW->filterChain->disengaged = false;

    //QSlider::mouseReleaseEvent(e);
}

void DisplaySlider::keyPressEvent(QKeyEvent *event)
{
    //QSlider::keyPressEvent(event);
}
