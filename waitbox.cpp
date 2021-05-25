#include "waitbox.h"
#include "mainwindow.h"

#include <QMovie>

WaitBox::WaitBox(QMainWindow *parent) : QDialog(parent)
{
    QLabel *text = new QLabel("Please wait while the model is loading...");
    QLabel *gif = new QLabel();
    QMovie *movie = new QMovie(":taurus.gif");
    gif->setMovie(movie);
    QGridLayout *layout = new QGridLayout();
    layout->addWidget(gif,      0, 0, 1, 1, Qt::AlignCenter);
    layout->addWidget(text,     1, 0, 1, 1, Qt::AlignCenter);
    setLayout(layout);
    setMinimumSize(QSize(300, 200));
    setWindowTitle("Darknet Model Loader");
    setModal(true);
    movie->start();
}
