#include "mainpanel.h"
#include "mainwindow.h"

MainPanel::MainPanel(QMainWindow *parent) : QWidget(parent)
{
    mainWindow = parent;

    //setStyleSheet("QWidget {background: blue; }");
    displayContainer = new DisplayContainer(mainWindow);
    //displayContainer->display->setStyleSheet(QString("QFrame { background-color: #32414B; padding: 0px; } "));
    displayContainer->display->setStyleSheet(QString("QFrame {background-color: %1; padding: 0px;}").arg(MW->config()->bm->color.name()));
    controlPanel = new ControlPanel(mainWindow);

    QGridLayout *layout = new QGridLayout();
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(5);
    layout->addWidget(displayContainer,   0, 0, 1, 1);
    layout->addWidget(controlPanel,       1, 0, 1, 1);

    layout->setRowStretch(0, 10);
    //layout->setRowStretch(4, 0);

    setLayout(layout);

}

void MainPanel::resizeEvent(QResizeEvent *event)
{

}
