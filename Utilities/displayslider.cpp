#include "displayslider.h"
#include "mainwindow.h"

DisplaySlider::DisplaySlider(QMainWindow *parent) : QSlider(parent)
{
    mainWindow = parent;

    //setOrientation(Qt::Horizontal);
    //setMinimum(0);
    //setMaximum(1000);
    //setValue(0);

    setMouseTracking(true);
    connect(this, SIGNAL(sliderMoved(int)), this, SLOT(moved(int)));
    connect(this, SIGNAL(sliderPressed()), this, SLOT(pressed()));
    connect(this, SIGNAL(sliderReleased()), this, SLOT(released()));
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

        const QPoint pos = mapToGlobal(QPoint(e->position().x(), geometry().top()));
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
    //QSlider::mouseReleaseEvent(e);

    if (!MW->e->looping)
        return;

    tick = 1000 * e->position().x() / width();

    SDL_Event event;
    SDL_memset(&event, 0, sizeof(event));
    event.type = MW->sdlCustomEventType;
    event.user.code = SLIDER_POSITION_UPDATE;
    event.user.data1 = &tick;
    SDL_PushEvent(&event);

}

void DisplaySlider::keyPressEvent(QKeyEvent *event)
{
    //QSlider::keyPressEvent(event);
}

void DisplaySlider::moved(int value)
{

}

void DisplaySlider::pressed()
{

}

void DisplaySlider::released()
{

}
