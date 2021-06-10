#include "viewer.h"
#include "mainwindow.h"

ViewerDialog::ViewerDialog(QMainWindow *parent) : PanelDialog(parent)
{
    mainWindow = parent;
    setWindowTitle("Viewer");
    viewer = new Viewer(mainWindow);
    QVBoxLayout *layout = new QVBoxLayout();
    layout->addWidget(viewer);
    setLayout(layout);
}

int ViewerDialog::getDefaultWidth()
{
    return defaultWidth;
}

int ViewerDialog::getDefaultHeight()
{
    return defaultHeight;
}

Viewer::Viewer(QMainWindow *parent) : QWidget(parent)
{
    mainWindow = parent;

    displayContainer = new DisplayContainer(mainWindow);
    displayContainer->display->setStyleSheet( "QLabel { background-color : lightGray; }" );

    QGridLayout *layout = new QGridLayout();
    layout->addWidget(displayContainer,   0, 0, 1, 1);
    setLayout(layout);
}
