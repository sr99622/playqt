#include "displayslider.h"
#include "mainwindow.h"

DisplaySlider::DisplaySlider(QMainWindow *parent)
{
    mainWindow = parent;

    setOrientation(Qt::Horizontal);
    setMinimum(0);
    setMaximum(1000);
    setValue(0);

    setTracking(true);
    connect(this, SIGNAL(sliderMoved(int)), this, SLOT(moved(int)));
    connect(this, SIGNAL(sliderPressed()), this, SLOT(pressed()));
    connect(this, SIGNAL(sliderReleased()), this, SLOT(released()));
}

void DisplaySlider::mouseReleaseEvent(QMouseEvent *e)
{
    QSlider::mouseReleaseEvent(e);
}

void DisplaySlider::keyPressEvent(QKeyEvent *event)
{
    QSlider::keyPressEvent(event);
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
