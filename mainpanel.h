#ifndef MAINPANEL_H
#define MAINPANEL_H

#include <QMainWindow>
#include <QLabel>

#include "controlpanel.h"

class MainPanel : public QWidget
{
    Q_OBJECT

public:
    MainPanel(QMainWindow *parent);

    QMainWindow *mainWindow;
    ControlPanel *controlPanel;
    QLabel *label;

};

#endif // MAINPANEL_H
