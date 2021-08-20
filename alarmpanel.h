#ifndef ALARMPANEL_H
#define ALARMPANEL_H

#include <QCheckBox>
#include <QLabel>
#include <QRunnable>
#include <QRadioButton>
#include <QSettings>
#include <QSlider>
#include <QFile>

#include "Utilities/kalman.h"
#include "Utilities/paneldialog.h"
#include "Utilities/numbertextbox.h"
#include "Utilities/filesetter.h"
#include "Utilities/directorysetter.h"
#include "configpanel.h"

#include <chrono>

using namespace std::chrono;

struct AudioData
{
    uint8_t *position;
    uint32_t length;
    int volume;
};

class AlarmPlayer : public QObject, public QRunnable
{
    Q_OBJECT

public:
    AlarmPlayer(QMainWindow *parent);
    void run() override;
    void stop();
    void mute(bool muted);
    static void audioCallback(void *userData, uint8_t *stream, int streamLength);

    QMainWindow *mainWindow;
    QString filename;
    AudioData audio;
    bool running = false;
    int lastVolume;

signals:
    void done();

};

class AlarmCheckBox : public QCheckBox
{
    Q_OBJECT

public:
    AlarmCheckBox(QString text, QString key, QSettings *settings, bool defaultState);

    QString key;
    QSettings *settings;

public slots:
    void clicked(bool);

};

class AlarmNumberBox : public NumberTextBox
{
    Q_OBJECT

public:
    AlarmNumberBox(QString key, QSettings *settings, QString defaultText);

    QString key;
    QSettings *settings;

public slots:
    void edited();

};

class AlarmPanel : public Panel
{
    Q_OBJECT

public:
    AlarmPanel(QMainWindow *parent, int obj_id);
    ~AlarmPanel();
    void autoSave() override;
    void feed(int count);
    QString key() const;
    QString getTimestampFilename() const;
    void writeToFile(const QString& str);

    int obj_id;
    AlarmNumberBox *minLimit;
    AlarmNumberBox *maxLimit;
    AlarmNumberBox *minLimitTime;
    AlarmNumberBox *maxLimitTime;
    AlarmCheckBox *chkMin;
    AlarmCheckBox *chkMax;
    AlarmCheckBox *chkSound;
    AlarmCheckBox *chkWrite;
    AlarmCheckBox *chkColor;
    FileSetter *soundSetter;
    DirectorySetter *dirSetter;
    QRadioButton *playOnce;
    QRadioButton *playContinuous;
    QPushButton *btnTest;
    QPushButton *btnMute;
    QSlider *volumeSlider;
    QLabel *filteredCount;
    QFile *file = nullptr;

    bool minAlarmOn = false;
    bool maxAlarmOn = false;
    high_resolution_clock::time_point minAlarmStartOn;
    high_resolution_clock::time_point minAlarmStartOff;
    high_resolution_clock::time_point maxAlarmStartOn;
    high_resolution_clock::time_point maxAlarmStartOff;
    high_resolution_clock::time_point reference;

    Kalman k_count;
    bool first_pass = true;
    high_resolution_clock::time_point lastTime;
    ColorProfile alarmProfile;
    ColorProfile storedProfile;
    AlarmPlayer *player;

    QString soundSetterKey;
    QString dirSetterKey;
    QString playOnceKey;
    QString volumeKey;

    bool testing = false;

public slots:
    void test();
    void setSoundPath(const QString&);
    void setDirPath(const QString&);
    void playOnceToggled(bool);
    void soundPlayFinished();
    void chkSoundClicked(bool);
    void chkColorClicked(bool);
    void chkWriteClicked(bool);
    void chkMinClicked(bool);
    void chkMaxClicked(bool);
    void volumeChanged(int);
    void minAlarmOff();
    void maxAlarmOff();
    void mute();

};

class AlarmDialog : public PanelDialog
{
    Q_OBJECT

public:
    AlarmDialog(QMainWindow *parent, int obj_id);
    AlarmPanel *getPanel();

};

#endif // ALARMPANEL_H
