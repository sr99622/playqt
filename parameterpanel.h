#ifndef PARAMETERPANEL_H
#define PARAMETERPANEL_H

#include <QMainWindow>
#include <QLabel>
#include <QComboBox>
#include <QLineEdit>
#include <QListWidget>
#include <QListWidgetItem>
#include <QSettings>
#include <QCheckBox>
#include <QMenu>

#include "Utilities/paneldialog.h"
#include "Ffplay/CommandOptions.h"

using namespace std;

class StoredOption : public QListWidgetItem
{
public:
    StoredOption(const QString& text);
    const QString toString();
    QString arg;
};

class SavedCmdLines : public QListWidget
{
    Q_OBJECT

public:
    SavedCmdLines(QMainWindow *parent);
    void keyPressEvent(QKeyEvent *event) override;
    void mouseDoubleClickEvent(QMouseEvent *event) override;

    QMainWindow *mainWindow;

public slots:
    void remove();
    void rename();

};

class ParameterPanel : public Panel
{
    Q_OBJECT

public:
    ParameterPanel(QMainWindow *parent);
    void setCmdLine();
    void addOptionToSaver(OptionDef option);
    const QString getOptionStorageString();
    void applyStyle();
    void saveSettings();
    void clearSettings();
    void restoreSettings();
    void set(int option_index, QString option_arg);

    QComboBox *options;
    QLineEdit *parameter;
    QLabel *cmdLineEquiv;
    QLineEdit *cmdLineName;
    SavedCmdLines *savedCmdLines;
    vector<OptionDef> saved_options;
    QMenu *menu;
    QString filters;

    const QString cmdKey = "ParameterPanel/savedCmdLine_";

public slots:
    void set();
    void clear();
    void apply();
    void optionChanged(int);
    void parameterEntered();
    void saveCmdLine();
    void itemChanged(QListWidgetItem*);
    void itemDoubleClicked(QListWidgetItem*);
    void showContextMenu(const QPoint&);
    void comboActivated(int);

};

class ParameterDialog : public PanelDialog
{
    Q_OBJECT

public:
    ParameterDialog(QMainWindow *parent);
    void show();

};

#endif // PARAMETERPANEL_H
