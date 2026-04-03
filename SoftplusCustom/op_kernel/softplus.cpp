#include "kernel_operator.h"

using namespace AscendC;

#define BUFFER_NUM 2  // 乒乓操作缓冲 buffer
#define DEBUG_ENABLE 1

class KernelSoftplus
{
public:
    __aicore__ inline KernelSoftplus() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                uint32_t tilingDataNum,

                                uint32_t bigCoreNum,
                                uint32_t smallCoreNum,
                                uint32_t bigCoreDataNum,
                                uint32_t smallCoreDataNum,
                                uint32_t bigCoreTailDataNum,
                                uint32_t smallCoreTailDataNum,
                                uint32_t bigCoreLoopNum,
                                uint32_t smallCoreLoopNum,

                                float32_t beta,
                                float32_t threshold)
    {
        int64_t coreIndex = AscendC::GetBlockIdx();
        uint32_t globalBufferIndex = bigCoreDataNum * coreIndex;  // 大核的数据地址偏移

        this->tilingDataNum = tilingDataNum;

        if (coreIndex < bigCoreNum)
        {
            this->loopNum = bigCoreLoopNum;
            this->coreDataNum = bigCoreDataNum;
            this->tailDataNum = bigCoreTailDataNum;
        }
        else
        {
            this->loopNum = smallCoreLoopNum;
            this->coreDataNum = smallCoreDataNum;
            this->tailDataNum = smallCoreTailDataNum;
            globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (coreIndex - bigCoreNum);  // 小核的数据地址偏移，每个大核比小核多出来的数据*当前是第几个小核
        }

        this->beta = beta;
        this->threshold = threshold;

        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + globalBufferIndex, this->coreDataNum);  // 计算每个核的第 0 轮数据处理的基础地址偏移
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + globalBufferIndex, this->coreDataNum);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tilingDataNum * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tilingDataNum * sizeof(DTYPE_Y));
        pipe.InitBuffer(calBuf, this->tilingDataNum * sizeof(float32_t));
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->loopNum;
        for (int32_t i = 0; i < loopCount; i++)
        {
            CopyIn(i, this->tilingDataNum);
            Compute(i, this->tilingDataNum);
            CopyOut(i, this->tilingDataNum);
        }
        if (this->tailDataNum != 0)
        {
            CopyIn(loopCount, this->tailDataNum);
            Compute(loopCount, this->tailDataNum);
            CopyOut(loopCount, this->tailDataNum);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t dataNum)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        // xGm 已经指向当前核负责数据段的起始地址，这里再通过 progress * tilingDataNum
        // 定位到当前核第 progress 轮要处理的 tile 起点，并搬运 dataNum 个元素到 UB。
        AscendC::DataCopy(xLocal, xGm[progress * this->tilingDataNum], dataNum);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int32_t progress, uint32_t dataNum)
    {
        auto scalar = 1;
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();

        if (std::is_same_v<DTYPE_X, bfloat16_t>)
        {
            AscendC::LocalTensor<float> tempTensor = calBuf.Get<float>(dataNum);
            
            AscendC::Cast(tempTensor, xLocal, AscendC::RoundMode::CAST_NONE, dataNum);
            AscendC::Muls(tempTensor, tempTensor, static_cast<float>(beta), dataNum);
            AscendC::Exp(tempTensor, tempTensor, dataNum);
            AscendC::Adds(tempTensor, tempTensor, static_cast<float>(scalar), dataNum);
            AscendC::Ln(tempTensor, tempTensor, dataNum);
            AscendC::Muls(tempTensor, tempTensor, static_cast<float>(1.0f / beta), dataNum);
            AscendC::Cast(yLocal, tempTensor, AscendC::RoundMode::CAST_RINT, dataNum);
        }
        else
        {
            AscendC::Muls(xLocal, xLocal, static_cast<DTYPE_X>(beta), dataNum);
            AscendC::Exp(xLocal, xLocal, dataNum);
            AscendC::Adds(xLocal, xLocal, static_cast<DTYPE_X>(scalar), dataNum);
            AscendC::Ln(xLocal, xLocal, dataNum);
            AscendC::Muls(yLocal, xLocal, static_cast<DTYPE_Y>(1.0f / beta), dataNum);
        }
        outQueueY.EnQue<DTYPE_Y>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress, uint32_t dataNum)
    {
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        AscendC::DataCopy(yGm[progress * this->tilingDataNum], yLocal, dataNum);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    // create queue for input, in this case depth is equal to buffer num
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    // create queue for output, in this case depth is equal to buffer num
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> calBuf;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;

    uint32_t tilingDataNum; // 单核单次 tiling 可处理的数据元素数
    uint32_t coreDataNum;   // 该核需要处理的总数居元素数
    uint32_t loopNum;       // 该核需要处理的循环次数，不包括最后一个尾处理
    uint32_t tailDataNum;   // 该核需要处理的尾数据元素数

    float32_t beta;
    float32_t threshold;
};

extern "C" __global__ __aicore__ void softplus(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelSoftplus op;
    op.Init(x, y,
            tiling_data.tilingDataNum,

            tiling_data.bigCoreNum,
            tiling_data.smallCoreNum,
            tiling_data.bigCoreDataNum,
            tiling_data.smallCoreDataNum,
            tiling_data.bigCoreTailDataNum,
            tiling_data.smallCoreTailDataNum,
            tiling_data.bigCoreLoopNum,
            tiling_data.smallCoreLoopNum,

            tiling_data.beta,
            tiling_data.threshold);
    op.Process();
}
