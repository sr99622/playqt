#ifndef CONTROLPANEL_H
#define CONTROLPANEL_H

#include <QMainWindow>
#include <QPushButton>
#include <QCheckBox>

#include "Frame.h"
#include "avexception.h"
#include "yolo_v2_class.hpp"

class ControlPanel : public QWidget
{
    Q_OBJECT

public:
    ControlPanel(QMainWindow *parent);

    QMainWindow *mainWindow;
    QCheckBox *engageFilter;
    AVExceptionHandler av;

public slots:
    void test();
    void pause();
    void mute();
    void quit();
    void fastforward();
    void rewind();

};

#endif // CONTROLPANEL_H
