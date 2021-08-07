#ifndef COUNTPANEL_H
#define COUNTPANEL_H

#include <QMainWindow>
#include <QListWidget>
#include <QTableWidget>
#include <QSplitter>
#include <QRadioButton>
#include <QGroupBox>
#include <QTimer>
#include <QFile>
#include <QMutex>
#include "Utilities/paneldialog.h"
#include "Utilities/directorysetter.h"
#include "Utilities/numbertextbox.h"
#include "Filters/darknet.h"
#include "alarmpanel.h"

class ObjDrawer : public QWidget
{
    Q_OBJECT

public:
    ObjDrawer(QMainWindow *parent, int obj_id);
    QString getButtonStyle() const;
    QString getSettingsKey() const;
    QString saveState() const;
    void restoreState(const QString& arg);
    void signalShown(int obj_id, const YUVColor& color);

    QMainWindow *mainWindow;
    QCheckBox *chkShow;
    QPushButton *btnColor;
    int obj_id;
    QColor color;
    bool show = false;

    const QString seperator = "\n";

signals:
    void shown(int, const YUVColor&);
    void colored(int, const YUVColor&);

public slots:
    void chkShowClicked(bool);
    void btnColorClicked();

};

class AlarmSetter : public QWidget
{
    Q_OBJECT

public:
    AlarmSetter(QMainWindow *parent, int obj_id);

    QMainWindow *mainWindow;
    QPushButton *button;
    int obj_id;

    AlarmDialog *alarmDialog;

public slots:
    void buttonPressed();

};

class CountPanel : public Panel
{
    Q_OBJECT

public:
    CountPanel(QMainWindow *parent);
    ~CountPanel();
    void autoSave() override;
    int indexForSums(int obj_id);
    int indexForCounts(int obj_id);
    int rowOf(int obj_id);
    int idFromName(const QString& name);
    void addNewLine(int obj_id);
    QString getTimestampFilename() const;
    void addCount(int obj_id, int count);

    QStringList names;
    QListWidget *list;
    QTableWidget *table;
    QSplitter *hSplit;
    DirectorySetter *dirSetter;
    NumberTextBox *txtInterval;
    QWidget *intervalPanel;
    QCheckBox *saveOn;
    QTimer *timer;
    QFile *file = nullptr;
    Darknet *darknet;

    const QString headerKey   = "CountPanel/header";
    const QString hSplitKey   = "CountPanel/hSplit";
    const QString dirKey      = "CountPanel/dir";
    const QString intervalKey = "CountPanel/interval";
    const QString groupBoxKey = "CountPanel/groupBox";
    const QString saveOnKey   = "CountPanel/saveOn";

    vector<pair<int, int>> sums;
    vector<pair<int, vector<int>>> counts;
    QMutex mutex;

public slots:
    void itemChanged(QListWidgetItem*);
    void itemClicked(QListWidgetItem*);
    void hSplitMoved(int, int);
    void feed(vector<bbox_t>*);
    void headerChanged(int, int, int);
    void setDir(const QString&);
    void saveOnClicked(bool);
    void intervalEdited();
    void timeout();

};

class CountDialog : public PanelDialog
{
    Q_OBJECT

public:
    CountDialog(QMainWindow *parent);

};

#endif // COUNTPANEL_H
