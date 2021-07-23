#ifndef FILEPANEL_H
#define FILEPANEL_H

#include <QMainWindow>
#include <QFileSystemModel>
#include <QTreeView>
#include <QHeaderView>
#include <QSettings>

#include "Utilities/directorysetter.h"
#include "Utilities/avexception.h"

class TreeView : public QTreeView
{

public:
    TreeView(QWidget *parent);
    void mouseDoubleClickEvent(QMouseEvent *event) override;

};

class FilePanel : public QWidget
{
    Q_OBJECT

public:
    FilePanel(QMainWindow *mainWindow, const QString& name, const QString& defaultPath);
    QString getDirKey() const;
    QString getHeaderKey() const;
    void autoSave();

    QMainWindow *mainWindow;
    DirectorySetter *directorySetter;
    QFileSystemModel *model;
    TreeView *tree;
    QMenu *menu;
    AVExceptionHandler av;
    QString name;
    QString defaultPath;
    bool changed = false;

signals:
    void msg(const QString&);

public slots:
    void setDirectory(const QString&);
    void doubleClicked(const QModelIndex&);
    void showContextMenu(const QPoint&);
    void remove();
    void rename();
    void info();
    void play();
    void headerChanged(int, int, int);

};

#endif // FILEPANEL_H
