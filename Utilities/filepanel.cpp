#include "filepanel.h"
#include "mainwindow.h"

FileTree::FileTree(QWidget *parent) : QTreeView(parent)
{
    panel = parent;
}

void FileTree::mouseDoubleClickEvent(QMouseEvent *event)
{
    ((FilePanel*)panel)->doubleClicked(indexAt(event->pos()));
}

void FileTree::keyPressEvent(QKeyEvent *event)
{
    switch(event->key()) {
    case Qt::Key_Return:
        ((FilePanel*)panel)->doubleClicked(currentIndex());
        break;
    case Qt::Key_Escape:
        SDL_Event sdl_event;
        sdl_event.type = FF_QUIT_EVENT;
        SDL_PushEvent(&sdl_event);
        break;
    case Qt::Key_Space:
        SDL_Event sdl_event_space;
        sdl_event_space.type = SDL_KEYDOWN;
        sdl_event_space.key.keysym.sym = SDLK_SPACE;
        SDL_PushEvent(&sdl_event_space);
        break;
    default:
        QTreeView::keyPressEvent(event);
        break;
    }
}

FilePanel::FilePanel(QMainWindow *parent) : QWidget(parent)
{
    mainWindow = parent;

    directorySetter = new DirectorySetter(mainWindow, "");
    directorySetter->trimHeight();
    model = new QFileSystemModel();
    model->setReadOnly(false);
    tree = new FileTree(this);
    tree->setModel(model);

    QGridLayout *layout = new QGridLayout();
    layout->addWidget(directorySetter,      0, 0, 1, 1);
    layout->addWidget(tree,                 1, 0, 1, 1);
    setLayout(layout);

    connect(directorySetter, SIGNAL(directorySet(const QString&)), this, SLOT(setDirectory(const QString&)));
    connect(this, SIGNAL(msg(const QString&)), mainWindow, SLOT(msg(const QString&)));

    menu = new QMenu("Context Menu", this);
    QAction *remove = new QAction("Delete", this);
    QAction *rename = new QAction("Rename", this);
    QAction *info = new QAction("Info", this);
    QAction *play = new QAction("Play", this);
    connect(remove, SIGNAL(triggered()), this, SLOT(remove()));
    connect(rename, SIGNAL(triggered()), this, SLOT(rename()));
    connect(info, SIGNAL(triggered()), this, SLOT(info()));
    connect(play, SIGNAL(triggered()), this, SLOT(play()));
    menu->addAction(remove);
    menu->addAction(rename);
    menu->addAction(info);
    menu->addAction(play);
    tree->setContextMenuPolicy(Qt::CustomContextMenu);
    connect(tree, SIGNAL(customContextMenuRequested(const QPoint&)), this, SLOT(showContextMenu(const QPoint&)));
}

void FilePanel::setDirectory(const QString& path)
{
    directorySetter->setPath(path);
    model->setRootPath(path);
    tree->setRootIndex(model->index(path));
}

void FilePanel::doubleClicked(const QModelIndex& index)
{
    if (index.isValid()) {
        //QString str = model->filePath(index);
        //MW->co->input_filename = av_strdup(str.toLatin1().data());
        //MW->runLoop();
        MW->mainPanel->controlPanel->play();
    }
}

void FilePanel::showContextMenu(const QPoint &pos)
{
    QModelIndex index = tree->indexAt(pos);
    if (index.isValid()) {
        menu->exec(mapToGlobal(pos));
    }
}

void FilePanel::remove()
{
    cout << "FilePanel::remove" << endl;
    QModelIndex index = tree->currentIndex();
    if (!index.isValid())
        return;

    int ret = QMessageBox::warning(this, "PlayQt",
                                   "You are about to delete this file.\n"
                                   "Are you sure you want to continue?",
                                   QMessageBox::Ok | QMessageBox::Cancel);

    if (ret == QMessageBox::Ok)
        QFile::remove(model->filePath(tree->currentIndex()).toLatin1().data());
}

void FilePanel::rename()
{
    QKeyEvent *event = new QKeyEvent(QEvent::KeyPress, Qt::Key_F2, Qt::NoModifier, QString("F2"));
    tree->keyPressEvent(event);
}
void FilePanel::info()
{
    cout << "FilePanel::info" << endl;
    QModelIndex index = tree->currentIndex();
    if (!index.isValid())
        return;

    QString filename = model->filePath(tree->currentIndex());
    AVFormatContext *fmt_ctx = nullptr;
    AVStream *video;
    AVStream *audio;
    int video_stream;
    int audio_stream;

    try {
        av.ck(avformat_open_input(&fmt_ctx, filename.toLatin1().data(), NULL, NULL), AOI);
        av.ck(avformat_find_stream_info(fmt_ctx, NULL), AFSI);
    }
    catch (AVException *e) {
        emit msg(QString("Unable to open format context %1: %2\n").arg(av.tag(e->cmd_tag), e->error_text));
    }

    try {
        av.ck(video_stream = av_find_best_stream(fmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0), AFBS);
        video = fmt_ctx->streams[video_stream];

        QString str = "File video parameters\n";

        char buf[16];

        QString codec_str;
        const AVCodecDescriptor *cd = avcodec_descriptor_get(video->codecpar->codec_id);
        if (cd) {
            QTextStream(&codec_str) << "codec_name: " << cd->name << "\n"
                                    << "codec_long_name: " << cd->long_name << "\n";
        }
        else {
            QTextStream(&codec_str) << "Uknown codec" << "\n";
        }

        if (fmt_ctx->metadata == NULL) {
            str.append("\nmetadata is NULL\n");
        }
        else {
            QTextStream(&str) << "\n";
            AVDictionaryEntry *t = NULL;
            while (t = av_dict_get(fmt_ctx->metadata, "", t, AV_DICT_IGNORE_SUFFIX)) {
                QTextStream(&codec_str) << t->key << " : " << t->value << "\n";
            }
        }

        QTextStream(&str)
            << "filename: " << filename << "\n"
            << "pixel format: " << av_get_pix_fmt_string(buf, 16, (AVPixelFormat)video->codecpar->format) << "\n"
            << codec_str
            << "format: " << fmt_ctx->iformat->long_name << " (" << fmt_ctx->iformat->name << ")\n"
            << "flags: " << fmt_ctx->iformat->flags << "\n"
            << "extradata_size: " << video->codecpar->extradata_size << "\n"
            << "codec time_base:  " << video->codec->time_base.num << " / " << video->codec->time_base.den << "\n"
            << "video time_base: " << video->time_base.num << " / " << video->time_base.den << "\n"
            << "codec framerate: " << video->codec->framerate.num << " / " << video->codec->framerate.den << "\n"
            << "gop_size: " << video->codec->gop_size << "\n"
            << "ticks_per_frame: " << video->codec->ticks_per_frame << "\n"
            << "bit_rate: " << fmt_ctx->bit_rate << "\n"
            << "codec framerate: " << av_q2d(video->codec->framerate) << "\n"
            << "start_time: " << fmt_ctx->start_time * av_q2d(av_get_time_base_q()) << "\n"
            << "duration: " << fmt_ctx->duration * av_q2d(av_get_time_base_q()) << "\n"
            << "width: " << video->codecpar->width << "\n"
            << "height: " << video->codecpar->height << "\n";

        emit msg(str);
        MW->messageBox->show();
    }
    catch (AVException *e) {
        emit msg(QString("Unable to process video stream %1: %2\n").arg(av.tag(e->cmd_tag), e->error_text));
    }

    try {
        av.ck(audio_stream = av_find_best_stream(fmt_ctx, AVMEDIA_TYPE_AUDIO, -1, -1, NULL, 0), AFBS);
        audio = fmt_ctx->streams[audio_stream];
        QString str = "File audio parameters\n";

        char buf[16];

        QString codec_str;
        const AVCodecDescriptor *cd = avcodec_descriptor_get(audio->codecpar->codec_id);
        if (cd) {
            QTextStream(&codec_str) << "codec_name: " << cd->name << "\n"
                                    << "codec_long_name: " << cd->long_name << "\n";
        }
        else {
            QTextStream(&codec_str) << "Uknown codec" << "\n";
        }

        if (fmt_ctx->metadata == NULL) {
            str.append("\nmetadata is NULL\n");
        }
        else {
            QTextStream(&str) << "\n";
            AVDictionaryEntry *t = NULL;
            while (t = av_dict_get(fmt_ctx->metadata, "", t, AV_DICT_IGNORE_SUFFIX)) {
                QTextStream(&codec_str) << t->key << " : " << t->value << "\n";
            }
        }

        QTextStream(&str)
            << "filename: " << filename << "\n"
            << codec_str

            << "format: " << fmt_ctx->iformat->long_name << " (" << fmt_ctx->iformat->name << ")\n"
            << "flags: " << fmt_ctx->iformat->flags << "\n"
            << "extradata_size: " << audio->codecpar->extradata_size << "\n"
            << "codec time_base:  " << audio->codec->time_base.num << " / " << audio->codec->time_base.den << "\n"
            << "audio time_base: " << audio->time_base.num << " / " << audio->time_base.den << "\n"
            << "codec framerate: " << audio->codec->framerate.num << " / " << audio->codec->framerate.den << "\n"
            << "ticks_per_frame: " << audio->codec->ticks_per_frame << "\n"
            << "bit_rate: " << fmt_ctx->bit_rate << "\n"
            << "codec framerate: " << av_q2d(audio->codec->framerate) << "\n"
            << "start_time: " << fmt_ctx->start_time * av_q2d(av_get_time_base_q()) << "\n"
            << "duration: " << fmt_ctx->duration * av_q2d(av_get_time_base_q()) << "\n";

        emit msg(str);
        MW->messageBox->show();
    }
    catch (AVException *e) {
        emit msg(QString("Unable to process audio stream %1: %2\n").arg(av.tag(e->cmd_tag), e->error_text));
    }

    if (fmt_ctx != nullptr)
        avformat_close_input(&fmt_ctx);
}
void FilePanel::play()
{
    doubleClicked(tree->currentIndex());
}
