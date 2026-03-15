#include "havoc/magics.hpp"

#include <array>
#include <vector>

namespace havoc::magics {

// ---------------------------------------------------------------------------
// Pre-computed magic numbers (from legacy magicsrands.h)
// ---------------------------------------------------------------------------

// clang-format off
static constexpr std::array<U64, 64> kBishopMagics = {
    U64(1143494273958400), U64(1155182134877503488), U64(4507998752866305),
    U64(1143775829164096), U64(432644906268229633), U64(143006304830720),
    U64(36173950270832640), U64(18176369755136), U64(316676545970432),
    U64(2269409188020736), U64(288247985526026240), U64(30872233312256),
    U64(18018848632471568), U64(9259401418526687232ULL), U64(551971471362),
    U64(2203326779392), U64(18296010992517248), U64(38280614147260480),
    U64(4503634526085152), U64(2251801994739976), U64(563018975936528),
    U64(281483570971136), U64(9288682829914112), U64(140742353817600),
    U64(4573968506290433), U64(2256232355201152), U64(52776591951936),
    U64(4620702013791944712), U64(2306406096639656000), U64(4616189892935028736),
    U64(282578850676992), U64(72199431046318080), U64(1143784151195648),
    U64(149671020594304), U64(1153484660718829696), U64(288265698500149376),
    U64(1126183374946568), U64(9011598375094272), U64(4507998747656704),
    U64(145146272350336), U64(36609348285237248), U64(290275633660928),
    U64(17867869263872), U64(9079695085696), U64(2305851814032966656),
    U64(4504767875252512), U64(565183353192960), U64(1134698149446912),
    U64(1163317729296384), U64(141322695475200), U64(140879357018112),
    U64(4503617914929152), U64(216172851940687872), U64(54052060609413120),
    U64(290499785465544704), U64(2260600206000136), U64(140877109411840),
    U64(1126466909701120), U64(547622928), U64(21107202),
    U64(18014398780023298), U64(2252074825876736), U64(4466833687552),
    U64(1130366677155968)
};

static constexpr std::array<U64, 64> kRookMagics = {
    U64(36029347045326849), U64(18016881537454080), U64(36037730551989376),
    U64(36037593380423296), U64(72062065099016192), U64(36030998189769728),
    U64(4647785733963121152), U64(36029072442132736), U64(140738562631808),
    U64(4899987038203285504), U64(4611827305940000768), U64(2305984296726317056),
    U64(141029554538496), U64(36169551687319681), U64(36169676249629184),
    U64(36169536654821632), U64(4539333759468672), U64(2314850484151664640),
    U64(282574769364992), U64(142386890018816), U64(2306969459010600964),
    U64(144397762631305218), U64(1154188691774636544), U64(142936520261697),
    U64(74379764743307296), U64(9024792514535496), U64(4620702014849941568),
    U64(8798247846016), U64(4512397868138880), U64(2341876206435041792),
    U64(565166156615688), U64(140739636380416), U64(18014948810559552),
    U64(36169809410400256), U64(2955555983335424), U64(598203061766400),
    U64(4512397893043200), U64(2306265238867018240), U64(1153062521276465664),
    U64(4399153809537), U64(140876001083410), U64(37225066206888064),
    U64(36310341252087872), U64(862017250427008), U64(567348134183040),
    U64(577028100336943232), U64(3378799265742976), U64(141014530654212),
    U64(288371123303751808), U64(4503874658369600), U64(70506184245504),
    U64(2542073031950464), U64(1134696067006592), U64(141991618936960),
    U64(140746080452736), U64(2392538392560000), U64(17871361048577),
    U64(70373040201857), U64(281750391490577), U64(17592723505157),
    U64(563019210294274), U64(281492156745729), U64(562952168081410),
    U64(1101661225986)
};
// clang-format on

// ---------------------------------------------------------------------------
// Unique-attack-count offsets per square (bishop and rook)
// ---------------------------------------------------------------------------

// clang-format off
static constexpr std::array<int, 64> kBishopOffsets = {
    7, 6, 10, 12, 12, 10, 6, 7,
    6, 6, 10, 12, 12, 10, 6, 6,
    10, 10, 40, 48, 48, 40, 10, 10,
    12, 12, 48, 108, 108, 48, 12, 12,
    12, 12, 48, 108, 108, 48, 12, 12,
    10, 10, 40, 48, 48, 40, 10, 10,
    6, 6, 10, 12, 12, 10, 6, 6,
    7, 6, 10, 12, 12, 10, 6, 7
};

static constexpr std::array<int, 64> kRookOffsets = {
    49, 42, 70, 84, 84, 70, 42, 49,
    42, 36, 60, 72, 72, 60, 36, 42,
    70, 60, 100, 120, 120, 100, 60, 70,
    84, 72, 120, 144, 144, 120, 72, 84,
    84, 72, 120, 144, 144, 120, 72, 84,
    70, 60, 100, 120, 120, 100, 60, 70,
    42, 36, 60, 72, 72, 60, 36, 42,
    49, 42, 70, 84, 84, 70, 42, 49
};
// clang-format on

// ---------------------------------------------------------------------------
// Internal tables
// ---------------------------------------------------------------------------

namespace detail {

/// Magic table entry for one square.
struct table {
    U64 mask = 0;
    U64 magic = 0;
    U16 offset = 0;
    U8 shift = 0;

    [[nodiscard]] unsigned entry(U64 occ) const {
        return static_cast<unsigned>(magic * (mask & occ) >> shift);
    }
};

// Index tables: magic-hash → unique-attack sub-index
static std::vector<std::vector<U8>> ridx(64, std::vector<U8>(4096));
static std::vector<std::vector<U8>> bidx(64, std::vector<U8>(512));

// Unique attack bitboards
static std::vector<U64> battks;
static std::vector<U64> rattks;

// Per-square magic lookup structs
static table rtable[64];
static table btable[64];

} // namespace detail

// ---------------------------------------------------------------------------
// Compute ray attacks for a given blocker configuration (init-time only)
// ---------------------------------------------------------------------------

namespace {

template <Piece p> U64 compute_attacks(Square s, U64 block) {
    constexpr int bishop_steps[] = {-7, 7, -9, 9};
    constexpr int rook_steps[] = {-1, 1, -8, 8};

    const int* steps = nullptr;
    int num_steps = 0;
    if constexpr (p == Piece::bishop) {
        steps = bishop_steps;
        num_steps = 4;
    } else {
        steps = rook_steps;
        num_steps = 4;
    }

    U64 bm = 0ULL;
    for (int i = 0; i < num_steps; ++i) {
        int step = steps[i];
        U64 tmp = 0ULL;
        int sqs = 1;

        while ((tmp & block) == 0ULL && sqs < 8) {
            int to = s + (sqs++) * step;
            if (on_board(to) && col_dist(s, to) <= 7 && row_dist(s, to) <= 7 &&
                ((p == Piece::bishop && on_diagonal(s, to)) ||
                 (p == Piece::rook && (same_row(s, to) || same_col(s, to))))) {
                tmp |= (1ULL << to);
            }
        }
        bm |= tmp;
    }
    return bm;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Magic table initialization — ported from legacy magics::load()
// ---------------------------------------------------------------------------

void init() {
    std::vector<U64> occupancy(4096);
    std::vector<U64> atks(4096);

    detail::battks.assign(1428, 0ULL);
    detail::rattks.assign(4900, 0ULL);

    for (Piece p = Piece::bishop; p <= Piece::rook; ++p) {
        for (Square s = Square::A1; s <= Square::H8; ++s) {
            U64 mask = (p == Piece::rook ? bitboards::rmask[s] : bitboards::bmask[s]);
            auto shift = static_cast<U8>(64 - bits::count(mask));
            U64 b = 0ULL;
            size_t occ_size = 0;

            // Enumerate all occupancy combinations of the mask
            do {
                occupancy[occ_size] = b;
                atks[occ_size++] = (p == Piece::bishop ? compute_attacks<Piece::bishop>(s, b)
                                                       : compute_attacks<Piece::rook>(s, b));
                b = (b - mask) & mask;
            } while (b);

            U16 offset = 0;
            U64 magic = (p == bishop ? kBishopMagics[s] : kRookMagics[s]);
            const auto& offsets = (p == bishop ? kBishopOffsets : kRookOffsets);

            for (size_t i = 0; i < static_cast<size_t>(s); ++i) {
                offset += static_cast<U16>(offsets[i]);
            }

            // Deduplicate attack patterns for this square
            std::array<U64, 144> stored = {};

            for (size_t i = 0; i < occ_size; ++i) {
                unsigned int idx = static_cast<unsigned>(magic * (mask & occupancy[i]) >> shift);
                U64 atk = atks[i];

                int k = 0;
                while (k < 144) {
                    U64 prev = stored[static_cast<size_t>(k)];
                    if (!prev && prev != atk) {
                        stored[static_cast<size_t>(k)] = atk;
                        break;
                    } else if (prev == atk) {
                        break;
                    }
                    ++k;
                }

                auto& indices = (p == bishop ? detail::bidx : detail::ridx);
                indices[s][idx] = static_cast<U8>(k);

                size_t o = static_cast<size_t>(indices[s][idx] + offset);
                auto& attack_arr = (p == bishop ? detail::battks : detail::rattks);
                attack_arr[o] = atks[i];

                detail::table* tab = (p == Piece::bishop ? detail::btable : detail::rtable);
                tab[s].magic = magic;
                tab[s].mask = mask;
                tab[s].shift = shift;
                tab[s].offset = offset;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Runtime attack lookup — template specializations
// ---------------------------------------------------------------------------

template <> U64 attacks<Piece::rook>(U64 occ, Square s) {
    using namespace detail;
    return rattks[ridx[s][rtable[s].entry(occ)] + rtable[s].offset];
}

template <> U64 attacks<Piece::bishop>(U64 occ, Square s) {
    using namespace detail;
    return battks[bidx[s][btable[s].entry(occ)] + btable[s].offset];
}

} // namespace havoc::magics
