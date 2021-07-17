#ifndef CONTROLPANEL_H
#define CONTROLPANEL_H

#include <QMainWindow>
#include <QPushButton>
#include <QCheckBox>
#include <QSlider>

//#include "Frame.h"
//#include "avexception.h"
//#include "yolo_v2_class.hpp"
#include "Utilities/displayslider.h"
#include "Utilities/avexception.h"

class ControlPanel : public QWidget
{
    Q_OBJECT

public:
    ControlPanel(QMainWindow *parent);
    void resizeEvent(QResizeEvent *event) override;
    bool checkCodec(const QString& filename);
    QString getButtonStyle(const QString& name) const;

    QMainWindow *mainWindow;
    QCheckBox *engageFilter;
    QSlider *volumeSlider;
    QPushButton *btnMute;
    QPushButton *btnPlay;
    QPushButton *btnStop;
    QPushButton *btnRewind;
    QPushButton *btnFastForward;
    QPushButton *btnNext;
    QPushButton *btnPrevious;
    QIcon icnAudioOn;
    QIcon icnAudioOff;
    QIcon icnPlay;
    QIcon icnPause;
    QIcon icnStop;
    QIcon icnRewind;
    QIcon icnFastForward;
    QIcon icnNext;
    QIcon icnPrevious;
    bool muted = false;
    bool paused = false;
    bool stopped = true;
    //bool input_switched = false;

    AVExceptionHandler av;

signals:
    void msg(const QString&);

public slots:
    void test();
    void play();
    void pause();
    void mute();
    void volup();
    void voldn();
    void quit();
    void fastforward();
    void rewind();
    void previous();
    void next();
    void singlestep();
    void engage(int);
    void sliderMoved(int);

};

#endif // CONTROLPANEL_H
