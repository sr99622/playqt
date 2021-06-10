#ifndef CONTROLPANEL_H
#define CONTROLPANEL_H

#include <QMainWindow>
#include <QPushButton>
#include <QCheckBox>

//#include "Frame.h"
//#include "avexception.h"
//#include "yolo_v2_class.hpp"
#include "Utilities/displayslider.h"

class ControlPanel : public QWidget
{
    Q_OBJECT

public:
    ControlPanel(QMainWindow *parent);
    void resizeEvent(QResizeEvent *event) override;

    QMainWindow *mainWindow;
    QCheckBox *engageFilter;
    DisplaySlider *slider;
    //AVExceptionHandler av;

public slots:
    void test();
    void pause();
    void mute();
    void volup();
    void voldn();
    void quit();
    void fastforward();
    void rewind();

};

#endif // CONTROLPANEL_H
