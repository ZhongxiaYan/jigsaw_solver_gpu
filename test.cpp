#include <opencv2/opencv.hpp>

#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>

using namespace std;
using namespace cv;

const int BLOCK_SIZE = 32;

random_device rd;
mt19937 rng(rd());

enum Direction {
    RIGHT = 0,
    DOWN = 1,
    LEFT = 2,
    UP = 3
};

const Point delta[4] = {Point(1, 0), Point(0, 1), Point(-1, 0), Point(0, -1)};

typedef pair<int, Direction> boundary;

// inclusive
inline int randint(int lo, int hi) {
    uniform_int_distribution<int> uni(lo, hi);
    return uni(rng);
}

// dissimilarity between two points
inline float dissimilarity(const Vec3f& v1, const Vec3f& v2) {
    float a = v1[0] - v2[0];
    float b = v1[1] - v2[1];
    float c = v1[2] - v2[2];
    return a * a + b * b + c * c;
}

// dissimilarity between two arrays of points
inline float dissimilarity(const Mat& m1, const Mat& m2) {
    assert(m1.total() == m2.total());
    float result = 0;
    MatConstIterator_<Vec3f> it1 = m1.begin<Vec3f>();
    MatConstIterator_<Vec3f> it1_end = m1.end<Vec3f>();
    MatConstIterator_<Vec3f> it2 = m2.begin<Vec3f>();
    while (it1 != it1_end) {
        result += dissimilarity(*it1++, *it2++);
    }
    return sqrt(result);
}

// argmin is expected to be called on 1-indexed things
int argmin(const Mat_<float>& m1) {
    int i = 0;
    int result = -1;
    float val = HUGE_VALF;
    for (const auto& f : m1) {
        if (f < val) {
            val = f;
            result = i;
        }
        i++;
    }
    assert(result > 0);
    return result;
}

int argmin(const Mat_<float>& m1, const bool* placed) {
    int i = 0;
    int result = -1;
    float val = HUGE_VALF;
    for (const auto& f : m1) {
        if (f < val && !placed[i]) {
            val = f;
            result = i;
        }
        i++;
    }
    assert(result > 0);
    return result;
}

void get_random_parent(int NUM_PIECES, int PUZZLE_HEIGHT, int PUZZLE_WIDTH, Mat_<int>& parent) {
    parent = Mat_<int>::zeros(PUZZLE_HEIGHT + 2, PUZZLE_WIDTH + 2);
    int* arr = new int[NUM_PIECES];
    int i;
    for (i = 0; i < NUM_PIECES; i++) {
        arr[i] = i + 1;
    }
    i = 0;
    shuffle(&arr[0], &arr[NUM_PIECES], rng);
    for (int y = 1; y <= PUZZLE_HEIGHT; y++) {
        for (int x = 1; x <= PUZZLE_WIDTH; x++) {
            parent(y, x) = arr[i++];
        }
    }
    delete[] arr;
}

// note: mat(y, x) == mat(Point(x, y))
// outputs: child, child_start
void crossover(int NUM_PIECES, int PUZZLE_HEIGHT, int PUZZLE_WIDTH,
               const Mat_<int>& parent1, const Mat_<int>& parent2, Mat_<int>& child, Point& child_start,
               const Point& parent1_start, const Point& parent2_start,
               const Mat_<float>& right_dissimilarity, const Mat_<float>& down_dissimilarity,
               const int* bb_left, const int* bb_right, const int* bb_up, const int* bb_down) {
    const float MUTATION_RATE = 0.05f;
    int x, y, i, j;
    child = Mat_<int>::zeros(2 * PUZZLE_HEIGHT + 1, 2 * PUZZLE_WIDTH + 1);
    Point* parent1_loc = new Point[NUM_PIECES + 1];
    Point* parent2_loc = new Point[NUM_PIECES + 1];
    Point* child_loc = new Point[NUM_PIECES + 1];
    bool* placed = new bool[NUM_PIECES + 1]();  // inits to false
    for (y = 0; y < PUZZLE_HEIGHT; y++) {
        for (x = 0; x < PUZZLE_WIDTH; x++) {
            auto a = Point(x, y) + parent1_start;
            auto b = Point(x, y) + parent2_start;
            parent1_loc[parent1(a)] = a;
            parent2_loc[parent2(b)] = b;
        }
    }
    int min_x, max_x, min_y, max_y;
    min_x = max_x = PUZZLE_WIDTH;
    min_y = max_y = PUZZLE_HEIGHT;
    bool maxed_height = false, maxed_width = false;
    int initial_piece = randint(1, NUM_PIECES);
    placed[initial_piece] = true;
    int remaining_count = NUM_PIECES - 1;
    child(PUZZLE_HEIGHT, PUZZLE_WIDTH) = initial_piece;
    child_loc[initial_piece] = Point(PUZZLE_WIDTH, PUZZLE_HEIGHT);
    vector<boundary> remaining_boundaries;
    remaining_boundaries.push_back(boundary(initial_piece, RIGHT));
    remaining_boundaries.push_back(boundary(initial_piece, DOWN));
    remaining_boundaries.push_back(boundary(initial_piece, LEFT));
    remaining_boundaries.push_back(boundary(initial_piece, UP));
    while (remaining_count > 0) {
        boundary b = remaining_boundaries.back();
        remaining_boundaries.pop_back();
        int p = b.first;
        Direction d = b.second;
        const Point& from = child_loc[p];
        Point to = from + delta[d];
        if (child(to)) continue;
        if (maxed_width && (to.x > max_x || to.x < min_x)) continue;
        if (maxed_height && (to.y > max_y || to.y < min_y)) continue;
        int p_new = 0;
        if (d == RIGHT) {
            p_new = argmin(right_dissimilarity.row(p), placed);
        } else if (d == DOWN) {
            p_new = argmin(down_dissimilarity.row(p), placed);
        } else if (d == LEFT) {
            p_new = argmin(right_dissimilarity.col(p), placed);
        } else if (d == UP) {
            p_new = argmin(down_dissimilarity.col(p), placed);
        }
        assert(p_new > 0);
        ////
        child(to) = p_new;
        child_loc[p_new] = to;
        placed[p_new] = true;
        min_x = min(min_x, to.x);
        max_x = max(max_x, to.x);
        min_y = min(min_y, to.y);
        max_y = max(max_y, to.y);
        assert(max_x - min_x < PUZZLE_WIDTH);
        assert(max_y - min_y < PUZZLE_HEIGHT);
        maxed_width = maxed_width || (max_x - min_x == PUZZLE_WIDTH - 1);
        maxed_height = maxed_height || (max_y - min_y == PUZZLE_HEIGHT - 1);
        if (!maxed_width) {
            remaining_boundaries.push_back(boundary(p_new, LEFT));
            remaining_boundaries.push_back(boundary(p_new, RIGHT));
        } else {
            if (to.x > min_x) {
                remaining_boundaries.push_back(boundary(p_new, LEFT));
            }
            if (to.x < max_x) {
                remaining_boundaries.push_back(boundary(p_new, RIGHT));
            }
        }
        if (!maxed_height) {
            remaining_boundaries.push_back(boundary(p_new, UP));
            remaining_boundaries.push_back(boundary(p_new, DOWN));
        } else {
            if (to.y > min_y) {
                remaining_boundaries.push_back(boundary(p_new, UP));
            }
            if (to.y < max_y) {
                remaining_boundaries.push_back(boundary(p_new, DOWN));
            }
        }
        shuffle(remaining_boundaries.begin(), remaining_boundaries.end(), rng);
        remaining_count--;
        ////
    }
    child_start.x = min_x;
    child_start.y = min_y;
    delete[] parent1_loc;
    delete[] parent2_loc;
    delete[] child_loc;
}

void reassemble(int PUZZLE_HEIGHT, int PUZZLE_WIDTH,
                const Mat_<int>& puzzle, const Point& puzzle_start,
                const Mat* pieces, Mat& result) {
    result.create(PUZZLE_HEIGHT * BLOCK_SIZE, PUZZLE_WIDTH * BLOCK_SIZE, pieces[1].type());
    for (int y = 0; y < PUZZLE_HEIGHT; y++) {
        for (int x = 0; x < PUZZLE_WIDTH; x++) {
            int p = puzzle(Point(x, y) + puzzle_start);
            pieces[p].copyTo(result(Rect(x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)));
        }
    }
}

void read_img(const char* path, Mat& result) {
    Mat img_bgr_u8 = imread(path);
    if (img_bgr_u8.empty()) {
        cerr << "unable to read input file" << endl;
        exit(1);
    }
    if (img_bgr_u8.type() != CV_8UC3) {
        cerr << "bad data format" << endl;
        exit(1);
    }
    Mat img_bgr_f;
    img_bgr_u8.convertTo(img_bgr_f, CV_32FC3, 1.0 / 255.0);
    Mat img_lab;
    cvtColor(img_bgr_f, img_lab, COLOR_BGR2Lab);
    Mat cropped = img_lab(Rect(0, 0, img_lab.cols - img_lab.cols % BLOCK_SIZE, img_lab.rows - img_lab.rows % BLOCK_SIZE));
    cropped.copyTo(result);
}

void write_img(const char* path, const Mat& img) {
    Mat img_bgr_f, img_bgr_u8;
    cvtColor(img, img_bgr_f, COLOR_Lab2BGR);
    img_bgr_f.convertTo(img_bgr_u8, CV_8UC3, 255);
    imwrite(path, img_bgr_u8);
}

int main(int argc, char* argv[]) {
    int i, j, x, y;
    const char* infile = argc > 1 ? argv[1] : "images/pillars.jpg";
    Mat img;
    read_img(infile, img);
    cout << "input size (cropped): " << img.size() << endl;
    const int PUZZLE_HEIGHT = img.rows / BLOCK_SIZE;
    const int PUZZLE_WIDTH = img.cols / BLOCK_SIZE;
    const int NUM_PIECES = PUZZLE_HEIGHT * PUZZLE_WIDTH;
    cout << NUM_PIECES << " pieces" << endl;
    // 1-indexed
    Mat* pieces = new Mat[NUM_PIECES + 1];
    i = 1;
    for (y = 0; y < PUZZLE_HEIGHT; y++) {
        for (x = 0; x < PUZZLE_WIDTH; x++) {
            pieces[i++] = img(Rect(x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE));
        }
    }
    Mat_<float> right_dissimilarity(NUM_PIECES + 1, NUM_PIECES + 1);
    Mat_<float> down_dissimilarity(NUM_PIECES + 1, NUM_PIECES + 1);
    // TODO: CUDA this?
    for (i = 0; i <= NUM_PIECES; i++) {
        for (j = 0; j <= NUM_PIECES; j++) {
            if (i == 0 || j == 0 || i == j) {
                right_dissimilarity(i, j) = HUGE_VALF;
                down_dissimilarity(i, j) = HUGE_VALF;
            } else {
                right_dissimilarity(i, j) = dissimilarity(pieces[i].col(pieces[i].cols - 1), pieces[j].col(0));
                down_dissimilarity(i, j) = dissimilarity(pieces[i].row(pieces[i].rows - 1), pieces[j].row(0));
            }
        }
    }
    // cout << format(cropped.at<Vec3f>(0, 0), Formatter::FMT_PYTHON) << endl;
    // cout << format(pieces[1], Formatter::FMT_PYTHON) << endl;
    // cout << format(pieces[2], Formatter::FMT_PYTHON) << endl;
    // cout << format(img_bgr, Formatter::FMT_PYTHON) << endl;
    // cout << right_dissimilarity(1, 2) << endl;
    // cout << down_dissimilarity(1, 2) << endl;
    int* best_left = new int[NUM_PIECES + 1];
    int* best_right = new int[NUM_PIECES + 1];
    int* bb_left = new int[NUM_PIECES + 1];
    int* bb_right = new int[NUM_PIECES + 1];
    int* best_up = new int[NUM_PIECES + 1];
    int* best_down = new int[NUM_PIECES + 1];
    int* bb_up = new int[NUM_PIECES + 1];
    int* bb_down = new int[NUM_PIECES + 1];
    for (i = 1; i <= NUM_PIECES; i++) {
        best_left[i] = argmin(right_dissimilarity.col(i));
        best_right[i] = argmin(right_dissimilarity.row(i));
        best_up[i] = argmin(down_dissimilarity.col(i));
        best_down[i] = argmin(down_dissimilarity.row(i));
    }
    for (i = 1; i <= NUM_PIECES; i++) {
        int bl = best_left[i], br = best_right[i];
        int bu = best_up[i], bd = best_down[i];
        bb_left[i] = best_right[bl] == i ? bl : 0;
        bb_right[i] = best_left[br] == i ? br : 0;
        bb_up[i] = best_down[bu] == i ? bu : 0;
        bb_down[i] = best_up[bd] == i ? bd : 0;
    }
    Mat_<int> p1, p2, child;
    Point child_start;
    get_random_parent(NUM_PIECES, PUZZLE_HEIGHT, PUZZLE_WIDTH, p1);
    get_random_parent(NUM_PIECES, PUZZLE_HEIGHT, PUZZLE_WIDTH, p2);
    crossover(NUM_PIECES, PUZZLE_HEIGHT, PUZZLE_WIDTH,
              p1, p2, child, child_start,
              Point(1, 1), Point(1, 1),
              right_dissimilarity, down_dissimilarity,
              bb_left, bb_right, bb_up, bb_down);
    // cout << format(child, Formatter::FMT_PYTHON) << endl;
    Mat result;
    reassemble(PUZZLE_HEIGHT, PUZZLE_WIDTH, child, child_start, pieces, result);
    if (result.size() != img.size()) {
        cout << "output size is different! this is bad, but continuing" << endl;
    }
    write_img("out.png", result);
    // don't bother freeing stuff allocated in main
    return 0;
}
