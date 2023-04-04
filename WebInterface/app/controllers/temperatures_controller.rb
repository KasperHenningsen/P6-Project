# frozen_string_literal: true
require 'will_paginate/array'

class TemperaturesController < ApplicationController
  before_action :set_temperature, only: %i[ show edit update destroy ]

  # GET /temperatures or /temperatures.json
  def index
    @temperatures = Temperature.order(date: :asc).paginate(page: params[:page], per_page: 500)
  end

  # GET /temperatures/1 or /temperatures/1.json
  def show
  end

  # GET /temperatures/new
  def new
    @temperature = Temperature.new
  end

  # GET /temperatures/1/edit
  def edit
  end

  # POST /temperatures or /temperatures.json
  def create
    @temperature = Temperature.new(temperature_params)

    respond_to do |format|
      if @temperature.save
        format.html { redirect_to temperature_url(@temperature), notice: "Temperature was successfully created." }
        format.json { render :show, status: :created, location: @temperature }
      else
        format.html { render :new, status: :unprocessable_entity }
        format.json { render json: @temperature.errors, status: :unprocessable_entity }
      end
    end
  end

  # PATCH/PUT /temperatures/1 or /temperatures/1.json
  def update
    respond_to do |format|
      if @temperature.update(temperature_params)
        format.html { redirect_to temperature_url(@temperature), notice: "Temperature was successfully updated." }
        format.json { render :show, status: :ok, location: @temperature }
      else
        format.html { render :edit, status: :unprocessable_entity }
        format.json { render json: @temperature.errors, status: :unprocessable_entity }
      end
    end
  end

  # DELETE /temperatures/1 or /temperatures/1.json
  def destroy
    @temperature.destroy

    respond_to do |format|
      format.html { redirect_to temperatures_url, notice: "Temperature was successfully destroyed." }
      format.json { head :no_content }
    end
  end

  private
    # Use callbacks to share common setup or constraints between actions.
    def set_temperature
      @temperature = Temperature.find(params[:id])
    end

    # Only allow a list of trusted parameters through.
    def temperature_params
      params.require(:temperature).permit(:date, :temp_min, :temp_max)
    end
end
