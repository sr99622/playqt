#ifndef FILEPANEL_H
#define FILEPANEL_H

#include <QMainWindow>
#include <QFileSystemModel>
#include <QTreeView>

#include "Utilities/directorysetter.h"

class FilePanel : public QWidget
{
    Q_OBJECT

public:
    FilePanel(QMainWindow *mainWindow);

    QMainWindow *mainWindow;
    DirectorySetter *directorySetter;
    QFileSystemModel *model;
    QTreeView *tree;

public slots:
    void setDirectory(const QString&);
    void doubleClicked(const QModelIndex&);
};

#endif // FILEPANEL_H
