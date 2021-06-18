#include "mainpanel.h"
#include "mainwindow.h"

MainPanel::MainPanel(QMainWindow *parent) : QWidget(parent)
{
    mainWindow = parent;

    displayContainer = new DisplayContainer(mainWindow);
    displayContainer->display->setStyleSheet("QFrame { background-color: #32414B; padding: 0px; } ");
    controlPanel = new ControlPanel(mainWindow);

    QGridLayout *layout = new QGridLayout();
    layout->addWidget(displayContainer,   0, 0, 4, 4);
    layout->addWidget(controlPanel,       4, 0, 4, 1);

    layout->setRowStretch(0, 10);
    layout->setRowStretch(4, 0);

    setLayout(layout);

}

void MainPanel::resizeEvent(QResizeEvent *event)
{

}
