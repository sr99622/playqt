#include "mainpanel.h"
#include "mainwindow.h"

MainPanel::MainPanel(QMainWindow *parent) : QWidget(parent)
{
    mainWindow = parent;

    controlPanel = new ControlPanel(mainWindow);
    label = new QLabel();
    label->setMinimumWidth(1280);
    label->setMinimumHeight(720);
    label->setStyleSheet("QLabel { background-color : lightGray; }");
    QGridLayout *layout = new QGridLayout();
    layout->addWidget(label,           0, 0, 4, 4);
    layout->addWidget(controlPanel,    4, 0, 4, 1);
    setLayout(layout);
}
