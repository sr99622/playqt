#ifndef VIEWER_H
#define VIEWER_H

#include <QMainWindow>
#include "Utilities/paneldialog.h"
#include "Utilities/displaycontainer.h"

class Viewer : public QWidget
{
    Q_OBJECT

public:
    Viewer(QMainWindow *parent);

    QMainWindow *mainWindow;
    DisplayContainer *displayContainer;

};

class ViewerDialog : public PanelDialog
{
    Q_OBJECT

public:
    ViewerDialog(QMainWindow *parent);
    int getDefaultWidth() override;
    int getDefaultHeight() override;

    QMainWindow *mainWindow;
    Viewer *viewer;

    int defaultWidth = 640;
    int defaultHeight = 480;
};

#endif // VIEWER_H
