#ifndef CONTROLPANEL_H
#define CONTROLPANEL_H

#include <QMainWindow>
#include <QPushButton>
#include <QCheckBox>
#include <QSlider>

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
    void styleButtons();
    void saveEngageSetting(bool arg);
    void restoreEngageSetting();

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

    const QString engageKey = "ControlPanel/engage";

    AVExceptionHandler av;

signals:
    void msg(const QString&);
    void quitting();
    void muting();

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
    void engage(bool);
    void sliderMoved(int);

};

#endif // CONTROLPANEL_H
