#ifndef FILEPANEL_H
#define FILEPANEL_H

#include <QMainWindow>
#include <QFileSystemModel>
#include <QTreeView>
#include <QSettings>

#include "Utilities/directorysetter.h"
#include "Utilities/avexception.h"

class FileTree : public QTreeView
{
    Q_OBJECT

public:
    FileTree(QWidget *parent);
    void mouseDoubleClickEvent(QMouseEvent *event) override;
    void keyPressEvent(QKeyEvent *event) override;

    QWidget *panel;
};

class FilePanel : public QWidget
{
    Q_OBJECT

public:
    FilePanel(QMainWindow *mainWindow);

    QMainWindow *mainWindow;
    DirectorySetter *directorySetter;
    QFileSystemModel *model;
    //FileTree *tree;
    QTreeView *tree;
    QMenu *menu;
    AVExceptionHandler av;

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
};

#endif // FILEPANEL_H
