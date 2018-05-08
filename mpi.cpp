#include <cassert>
#include <cmath>
#include <cstdlib>
#include <string>
#include <iostream>
#include <sstream>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include <tuple>
#include <utility>
#include <omp.h>
#include <mpi.h>
#include <sys/stat.h>

#include <boost/container/flat_set.hpp>
#include <boost/program_options.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

constexpr int BLOCK_SIZE = 32;
int NUM_PIECES, PUZZLE_HEIGHT, PUZZLE_WIDTH;

mt19937 rng(random_device{}());

enum Direction {
    // exact order is important
    RIGHT = 0,
    DOWN = 1,
    LEFT = 2,
    UP = 3
};

const Point delta[4] = {Point(1, 0), Point(0, 1), Point(-1, 0), Point(0, -1)};

int* bb_by_dir[4];

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

// sum dissimilarity of entire puzzle
float dissimilarity(const Mat_<int>& puzzle, const Point& puzzle_start,
                    const Mat_<float>& right_dissimilarity, const Mat_<float>& down_dissimilarity) {
    float result = 0;
    for (int y = 0; y < PUZZLE_HEIGHT - 1; y++) {
        for (int x = 0; x < PUZZLE_WIDTH - 1; x++) {
            const Point pt = puzzle_start + Point(x, y);
            int p = puzzle(pt);
            int p_r = puzzle(pt + delta[RIGHT]);
            int p_d = puzzle(pt + delta[DOWN]);
            assert(p && p_r && p_d);
            result += right_dissimilarity(p, p_r) + down_dissimilarity(p, p_d);
        }
    }
    return result;
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

int argmin(const Mat_<float>& m1, const boost::container::flat_set<int>& unplaced) {
    int result = -1;
    float val = HUGE_VALF;
    for (const auto& p : unplaced) {
        float f = m1.at<float>(p);
        if (f < val) {
            val = f;
            result = p;
        }
    }
    assert(result > 0);
    return result;
}

void get_random_parent(Mat_<int>& result) {
    result = Mat_<int>::zeros(PUZZLE_HEIGHT + 2, PUZZLE_WIDTH + 2);
    int* arr = new int[NUM_PIECES];
    int i;
    for (i = 0; i < NUM_PIECES; i++) {
        arr[i] = i + 1;
    }
    i = 0;
    shuffle(arr, arr + NUM_PIECES, rng);
    for (int y = 1; y <= PUZZLE_HEIGHT; y++) {
        for (int x = 1; x <= PUZZLE_WIDTH; x++) {
            result(y, x) = arr[i++];
        }
    }
    delete[] arr;
}

void get_solution(Mat_<int>& result) {
    result = Mat_<int>::zeros(PUZZLE_HEIGHT + 2, PUZZLE_WIDTH + 2);
    int i = 1;
    for (int y = 1; y <= PUZZLE_HEIGHT; y++) {
        for (int x = 1; x <= PUZZLE_WIDTH; x++) {
            result(y, x) = i++;
        }
    }
}

// note: mat(y, x) == mat(Point(x, y))
// outputs: child, child_start
void crossover(const Mat_<int>& parent1, const Mat_<int>& parent2, Mat_<int>& child, Point& child_start,
               const Point& parent1_start, const Point& parent2_start,
               const Mat_<float>& right_dissimilarity, const Mat_<float>& down_dissimilarity) {
    constexpr int MUTATION_RATE = 1;
    constexpr int MUTATION_RATE_MAX = 1000;
    int x, y;
    child = Mat_<int>::zeros(2 * PUZZLE_HEIGHT + 1, 2 * PUZZLE_WIDTH + 1);
    Point* parent1_loc = new Point[NUM_PIECES + 1];
    Point* parent2_loc = new Point[NUM_PIECES + 1];
    Point* child_loc = new Point[NUM_PIECES + 1];
    boost::container::flat_set<int> unplaced;
    for (int i = 1; i <= NUM_PIECES; i++) {
        unplaced.insert(i);
    }
    int* tmp = new int[NUM_PIECES];
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
    unplaced.erase(initial_piece);
    int remaining_count = NUM_PIECES - 1;
    child(PUZZLE_HEIGHT, PUZZLE_WIDTH) = initial_piece;
    child_loc[initial_piece] = Point(PUZZLE_WIDTH, PUZZLE_HEIGHT);

    vector<boundary> remaining_boundaries;
    vector<pair<int, Point>> bb_candidates;

    auto get_random_unplaced = [&]() {
        assert(unplaced.size() == remaining_count);
        return unplaced.begin()[randint(0, remaining_count - 1)];
    };

    auto append_and_shuffle = [&](const boundary& b) {
        size_t i = randint(0, remaining_boundaries.size());
        if (i == remaining_boundaries.size()) {
            remaining_boundaries.push_back(b);
        } else {
            remaining_boundaries.push_back(remaining_boundaries[i]);
            remaining_boundaries[i] = b;
        }
    };

    auto add_piece = [&](int p, const Point& to) {
        assert(p > 0 && p <= NUM_PIECES);
        assert(unplaced.find(p) != unplaced.end());
        child(to) = p;
        child_loc[p] = to;
        unplaced.erase(p);
        min_x = min(min_x, to.x);
        max_x = max(max_x, to.x);
        min_y = min(min_y, to.y);
        max_y = max(max_y, to.y);
        assert(max_x - min_x < PUZZLE_WIDTH);
        assert(max_y - min_y < PUZZLE_HEIGHT);
        maxed_width = maxed_width || (max_x - min_x == PUZZLE_WIDTH - 1);
        maxed_height = maxed_height || (max_y - min_y == PUZZLE_HEIGHT - 1);
        if (!maxed_width) {
            append_and_shuffle(boundary(p, LEFT));
            append_and_shuffle(boundary(p, RIGHT));
        } else {
            if (to.x > min_x) {
                append_and_shuffle(boundary(p, LEFT));
            }
            if (to.x < max_x) {
                append_and_shuffle(boundary(p, RIGHT));
            }
        }
        if (!maxed_height) {
            append_and_shuffle(boundary(p, UP));
            append_and_shuffle(boundary(p, DOWN));
        } else {
            if (to.y > min_y) {
                append_and_shuffle(boundary(p, UP));
            }
            if (to.y < max_y) {
                append_and_shuffle(boundary(p, DOWN));
            }
        }
        remaining_count--;
    };

    append_and_shuffle(boundary(initial_piece, RIGHT));
    append_and_shuffle(boundary(initial_piece, DOWN));
    append_and_shuffle(boundary(initial_piece, LEFT));
    append_and_shuffle(boundary(initial_piece, UP));

    while (remaining_count > 0) {
        bool do_continue = false;
        bb_candidates.clear();
        for (size_t i = 0; i < remaining_boundaries.size(); i++) {
            // don't delete elements in this loop (at least for now)
            const boundary& b = remaining_boundaries[i];
            int p = b.first;
            const Direction& d = b.second;
            const Point& from = child_loc[p];
            ////
            const Point to = from + delta[d];
            if (child(to)) continue;
            if (maxed_width && (to.x > max_x || to.x < min_x)) continue;
            if (maxed_height && (to.y > max_y || to.y < min_y)) continue;
            ////
            const Point to_p1 = parent1_loc[p] + delta[d];
            const Point to_p2 = parent2_loc[p] + delta[d];
            int p1 = parent1(to_p1);
            int p2 = parent2(to_p2);
            if (p1 && p1 == p2 && (unplaced.find(p1) != unplaced.end())) {
                if (remaining_count > 1 && randint(0, MUTATION_RATE_MAX - 1) < MUTATION_RATE) {
                    p1 = get_random_unplaced();
                }
                add_piece(p1, to);
                do_continue = true;
                break;
            } else {
                int bb = bb_by_dir[d][p];
                if (bb) {
                    if (p1 == bb && (unplaced.find(p1) != unplaced.end())) {
                        bb_candidates.push_back(make_pair(p1, to));
                    }
                    if (p2 == bb && (unplaced.find(p2) != unplaced.end())) {
                        bb_candidates.push_back(make_pair(p2, to));
                    }
                }
            }
        }
        if (do_continue) continue;
        if (bb_candidates.size() > 0) {
            const auto& pto = bb_candidates[randint(0, bb_candidates.size() - 1)];
            add_piece(pto.first, pto.second);
            continue;
        }
        const boundary& b = remaining_boundaries.back();
        remaining_boundaries.pop_back();
        int p = b.first;
        const Direction& d = b.second;
        const Point& from = child_loc[p];
        ////
        const Point to = from + delta[d];
        if (child(to)) continue;
        if (maxed_width && (to.x > max_x || to.x < min_x)) continue;
        if (maxed_height && (to.y > max_y || to.y < min_y)) continue;
        ////
        int p_new = 0;
        if (d == RIGHT) {
            p_new = argmin(right_dissimilarity.row(p), unplaced);
        } else if (d == DOWN) {
            p_new = argmin(down_dissimilarity.row(p), unplaced);
        } else if (d == LEFT) {
            p_new = argmin(right_dissimilarity.col(p), unplaced);
        } else if (d == UP) {
            p_new = argmin(down_dissimilarity.col(p), unplaced);
        }
        if (remaining_count > 1 && randint(0, MUTATION_RATE_MAX - 1) < MUTATION_RATE) {
            p_new = get_random_unplaced();
        }
        add_piece(p_new, to);
    }
    child_start = Point(min_x, min_y);
    delete[] parent1_loc;
    delete[] parent2_loc;
    delete[] child_loc;
    delete[] tmp;
}

void reassemble(const Mat_<int>& puzzle, const Point& puzzle_start,
                const Mat* pieces, Mat& result) {
    result.create(PUZZLE_HEIGHT * BLOCK_SIZE, PUZZLE_WIDTH * BLOCK_SIZE, pieces[1].type());
    for (int y = 0; y < PUZZLE_HEIGHT; y++) {
        for (int x = 0; x < PUZZLE_WIDTH; x++) {
            int p = puzzle(Point(x, y) + puzzle_start);
            // if (!p) {
            //     cout << puzzle_start << endl;
            //     cout << Point(x, y) << endl;
            //     cout << format(puzzle, Formatter::FMT_PYTHON) << endl;
            // }
            assert(p);
            pieces[p].copyTo(result(Rect(x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)));
        }
    }
}

void read_img(const string& path, Mat& result) {\
    cout << "reading image from " << path << endl;
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

void write_img(const string& path, const Mat& img) {
    Mat img_bgr_f, img_bgr_u8;
    cvtColor(img, img_bgr_f, COLOR_Lab2BGR);
    img_bgr_f.convertTo(img_bgr_u8, CV_8UC3, 255);
    imwrite(path, img_bgr_u8);
}

int main(int argc, char* argv[]) {
    auto start = chrono::steady_clock::now();

    boost::program_options::options_description desc;
    desc.add_options()
        ("help,h", "show help")
        ("threads,t", boost::program_options::value<string>()->default_value("0"), "number of threads on each mpi node (0 for auto, comma seperated)")
        ("generations,g", boost::program_options::value<int>()->default_value(100), "number of generations")
        ("populations,p", boost::program_options::value<string>()->default_value("1000"), "population size on each mpi node (comma separated)")
        ("out,o", boost::program_options::value<string>()->default_value("."), "output directory")
        ("in,i", boost::program_options::value<string>()->default_value("images/pillars.jpg"), "input image")
        ;
    boost::program_options::variables_map vm;
    boost::program_options::positional_options_description p_desc;
    p_desc.add("in", -1);
    boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(desc).positional(p_desc).run(), vm);
    boost::program_options::notify(vm);

    if (vm.count("help") > 0) {
        cerr << desc << endl;
        cerr << "note: --in/-i not needed before input filename" << endl;
        return 1;
    }
    std::stringstream threads_s(vm["threads"].as<string>());
    std::stringstream populations_s(vm["populations"].as<string>());
    const int generations = vm["generations"].as<int>();

    int n_proc, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &n_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int i_proc = 0;
    int threads;
    int population_size;
    while (threads_s >> threads && populations_s >> population_size) {
        if (i_proc == rank) break;
        threads_s.ignore();
        populations_s.ignore();
        i_proc++;
    }

    if (threads > 0) {
        cout << "using " << threads << " threads" << endl;
        omp_set_num_threads(threads);
    } else {
        cout << "using default threads" << endl;
    }
    constexpr int NUM_ELITE = 4;
    const int to_add = population_size - NUM_ELITE;
    const string outdir = vm["out"].as<string>();
    mkdir(outdir.c_str(), 0777);
    cout << "generations: " << generations << ", population size: " << population_size << endl;

    int i, j, x, y;
    Mat img;

    read_img(vm["in"].as<string>(), img);
    cout << "input size (cropped): " << img.size() << endl;
    // cout << format(img.at<Vec3f>(0, 0), Formatter::FMT_PYTHON) << endl;
    PUZZLE_HEIGHT = img.rows / BLOCK_SIZE;
    PUZZLE_WIDTH = img.cols / BLOCK_SIZE;
    NUM_PIECES = PUZZLE_HEIGHT * PUZZLE_WIDTH;
    assert(PUZZLE_HEIGHT > 1 && PUZZLE_WIDTH > 1);
    cout << NUM_PIECES << " pieces" << endl;

    // 1-indexed
    Mat* pieces = new Mat[NUM_PIECES + 1];
    i = 1;
    for (y = 0; y < PUZZLE_HEIGHT; y++) {
        for (x = 0; x < PUZZLE_WIDTH; x++) {
            pieces[i++] = img(Rect(x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE));
        }
    }

    cout << "precomputing pairwise dissimilarities" << endl;
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

    cout << "precomputing best buddies" << endl;
    int* best_left = new int[NUM_PIECES + 1];
    int* best_right = new int[NUM_PIECES + 1];
    int* best_up = new int[NUM_PIECES + 1];
    int* best_down = new int[NUM_PIECES + 1];
    int* bb_left = new int[NUM_PIECES + 1];
    int* bb_right = new int[NUM_PIECES + 1];
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
    bb_by_dir[RIGHT] = bb_right;
    bb_by_dir[DOWN] = bb_down;
    bb_by_dir[LEFT] = bb_left;
    bb_by_dir[UP] = bb_up;

    Mat_<int> parent;
    Mat_<int> child;
    Point child_start;
    Mat assembled_result;

    cout << "creating initial population" << endl;
    vector<tuple<Mat_<int>, Point, float>> population;
    vector<tuple<Mat_<int>, Point, float>> new_population;
    auto cmp = [](const auto& a, const auto& b) {
        return get<2>(a) < get<2>(b);
    };

    for (i = 0; i < population_size; i++) {
        get_random_parent(parent);
        population.push_back(make_tuple(parent.clone(), Point(1, 1),
                             dissimilarity(parent, Point(1, 1), right_dissimilarity, down_dissimilarity)));
    }

    Mat_<int> solution;
    get_solution(solution);
    cout << "solution loss: " << dissimilarity(solution, Point(1, 1), right_dissimilarity, down_dissimilarity) << endl;
    cout << "initialization time: " << chrono::duration <double, milli> (chrono::steady_clock::now() - start).count() << " ms" << endl;
    start = chrono::steady_clock::now();
    for (i = 1; i <= generations; i++) {
        nth_element(population.begin(), population.begin() + NUM_ELITE, population.end(), cmp);
        new_population.assign(population.begin(), population.begin() + NUM_ELITE);
        #pragma omp parallel for private(child, child_start)
        for (j = 0; j < to_add; j++) {
            int idx1 = randint(0, population_size - 1);
            int idx2;
            do {
                idx2 = randint(0, population_size - 1);
            } while (idx1 == idx2);
            const auto& pop1 = population[idx1];
            const auto& pop2 = population[idx2];
            crossover(get<0>(pop1), get<0>(pop2), child, child_start,
                      get<1>(pop1), get<1>(pop2),
                      right_dissimilarity, down_dissimilarity);
            #pragma omp critical
            {
                new_population.push_back(make_tuple(child.clone(), child_start,
                                         dissimilarity(child, child_start, right_dissimilarity, down_dissimilarity)));
            }
        }
        assert(new_population.size() == population_size);
        population = move(new_population);
        cout << "finished generation " << i << "; time=" << chrono::duration <double, milli> (chrono::steady_clock::now() - start).count() << " ms" << endl;
        const auto& best = *min_element(population.begin(), population.end(), cmp);
        // using std::tie(child, child_start, ...) leads to weird bugs here
        cout << "best loss: " << get<2>(best) << endl;
        reassemble(get<0>(best), get<1>(best), pieces, assembled_result);
        // cout << format(child, Formatter::FMT_PYTHON) << endl;
        write_img(outdir + "/out" + to_string(i) + ".png", assembled_result);
    }
    // don't bother freeing stuff allocated in main
    return 0;
}
