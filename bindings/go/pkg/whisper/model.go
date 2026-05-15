package whisper

import (
	"fmt"
	"os"
	"runtime"

	// Bindings
	whisper "github.com/ggerganov/whisper.cpp/bindings/go"
)

///////////////////////////////////////////////////////////////////////////////
// TYPES

type model struct {
	path string
	ctx  *whisper.Context
}

// Make sure model adheres to the interface
var _ Model = (*model)(nil)

type SamplingStrategy whisper.SamplingStrategy

const (
	SAMPLING_GREEDY      SamplingStrategy = (SamplingStrategy)(whisper.SAMPLING_GREEDY)
	SAMPLING_BEAM_SEARCH SamplingStrategy = (SamplingStrategy)(whisper.SAMPLING_BEAM_SEARCH)
)

///////////////////////////////////////////////////////////////////////////////
// LIFECYCLE

// New creates a new whisper model with default parameters (CPU-only)
func New(path string) (Model, error) {
	return NewWithParams(path, false, 0, false)
}

// NewWithGPU creates a new whisper model with GPU enabled
func NewWithGPU(path string, gpuDevice int, flashAttn bool) (Model, error) {
	return NewWithParams(path, true, gpuDevice, flashAttn)
}

// NewWithParams creates a new whisper model with custom parameters
func NewWithParams(path string, useGPU bool, gpuDevice int, flashAttn bool) (Model, error) {
	model := new(model)
	if _, err := os.Stat(path); err != nil {
		return nil, err
	}

	// Create context parameters with GPU settings
	params := whisper.Whisper_context_default_params()
	params.SetUseGPU(useGPU)
	params.SetGPUDevice(gpuDevice)
	params.SetFlashAttn(flashAttn)

	// Initialize context with parameters
	ctx := whisper.Whisper_init_from_file_with_params(path, params)
	if ctx == nil {
		return nil, ErrUnableToLoadModel
	}

	model.ctx = ctx
	model.path = path

	// Return success
	return model, nil
}

func (model *model) Close() error {
	if model.ctx != nil {
		model.ctx.Whisper_free()
	}

	// Release resources
	model.ctx = nil

	// Return success
	return nil
}

///////////////////////////////////////////////////////////////////////////////
// STRINGIFY

func (model *model) String() string {
	str := "<whisper.model"
	if model.ctx != nil {
		str += fmt.Sprintf(" model=%q", model.path)
	}
	return str + ">"
}

///////////////////////////////////////////////////////////////////////////////
// PUBLIC METHODS

// Return true if model is multilingual (language and translation options are supported)
func (model *model) IsMultilingual() bool {
	return model.ctx.Whisper_is_multilingual() != 0
}

// Return all recognized languages. Initially it is set to auto-detect
func (model *model) Languages() []string {
	result := make([]string, 0, whisper.Whisper_lang_max_id())
	for i := 0; i < whisper.Whisper_lang_max_id(); i++ {
		str := whisper.Whisper_lang_str(i)
		if model.ctx.Whisper_lang_id(str) >= 0 {
			result = append(result, str)
		}
	}
	return result
}

func (model *model) NewContext() (Context, error) {
	// By default, specify the greedy strategy
	return model.NewContextWithStrategy(SAMPLING_GREEDY)
}

func (model *model) NewContextWithStrategy(strategy SamplingStrategy) (Context, error) {
	if model.ctx == nil {
		return nil, ErrInternalAppError
	}

	// Create new context
	params := model.ctx.Whisper_full_default_params((whisper.SamplingStrategy)(strategy))
	params.SetTranslate(false)
	params.SetPrintSpecial(false)
	params.SetPrintProgress(false)
	params.SetPrintRealtime(false)
	params.SetPrintTimestamps(false)
	params.SetThreads(runtime.NumCPU())
	params.SetNoContext(true)

	// Return new context
	return newContext(model, params)
}
