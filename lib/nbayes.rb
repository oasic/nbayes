require 'yaml'
require_relative 'dbconnection'

module NBayes
  class Vocab
    attr_accessor :log_size, :tokens

    def initialize(options = {})
      @tokens = DBToken.new
      @log_size = options[:log_size]
    end

    def delete(token)
      tokens.delete(token)
    end

    def each(&block)
      tokens.keys.each(&block)
    end

    def size
      if log_size
        Math.log(tokens.count)
      else
        tokens.count
      end
    end

    def seen_token(token, category)
      tokens.update_frequency(token, 1, category)
    end
  end

  class Data
    attr_accessor :data

    def initialize(options = {})
      @data = DBData.new
    end

    def categories
      data.keys
    end

    def token_trained?(token, category)
      data[category] ? data[category][:tokens].has_key?(token) : false
    end

    def cat_data(category)
      unless data[category].is_a? Hash
        data[category] = new_category
      end
      data[category]
    end

    def category_stats
      tmp = []
      total_example_count = total_examples
      self.each do |category|
        e = example_count(category)
        t = token_count(category)
        tmp << "For category #{category}, %d examples (%.02f%% of the total) and %d total_tokens" % [e, 100.0 * e / total_example_count, t]
      end
      tmp.join("\n")
    end

    def each(&block)
      data.keys.each(&block)
    end

    def example_count(category)
      cat_data(category)[:examples]
    end

    def token_count(category)
      cat_data(category)[:total_tokens]
    end

    def total_examples
      sum = 0
      self.each {|category| sum += example_count(category) }
      sum
    end

    def add_token_to_category(category, token)
      data.upsert(category, token)
    end

    def remove_token_from_category(category, token)
      data.remove_from_category(category, token)
      delete_token_from_category(category, token) if cat_data(category)[:tokens][token] < 1
      delete_category(category) if cat_data(category)[:total_tokens] < 1
    end

    def count_of_token_in_category(category, token)
      count = cat_data(category)[:tokens][token]
      count.nil? ? 0 : count
    end

    def delete_token_from_category(category, token)
      data.delete_from_category(category, token)
    end

    def new_category
      {
        :tokens => Hash.new(0),
        :total_tokens => 0,
        :examples => 0
      }
    end

    def delete_category(category)
      data.delete(category) if data.has_key?(category)
      categories
    end

  end

  class Base
    attr_accessor :assume_uniform, :debug, :k, :vocab, :data
    attr_reader :binarized

    def initialize(options={})
      @debug = false
      @k = 1
      @binarized = options[:binarized] || false
      @assume_uniform = false
      @vocab = Vocab.new(:log_size => options[:log_vocab])
      @data = Data.new
    end

    def purge_less_than(x)
      remove_list = {}
      @vocab.each do |token|
        if data.purge_less_than(token, x)
          remove_list[token] = 1
        end
      end
      remove_list.keys.each {|token| @vocab.delete(token) }
    end

    def delete_category(category)
      data.delete_category(category)
    end

    def train(tokens, category)
      tokens = tokens.uniq if binarized
      tokens.each do |token|
        data.add_token_to_category(category, token)
      end
    end

    def untrain(tokens, category)
      tokens = tokens.uniq if binarized
      tokens.each do |token|
        if data.token_trained?(token, category)
          vocab.delete(token)
          data.remove_token_from_category(category, token)
        end
      end
    end

    def classify(tokens)
      probs = {}
      tokens = tokens.uniq if binarized
      probs = calculate_probabilities(tokens)
      probs.extend(NBayes::Result)
      probs
    end

    def category_stats
      data.category_stats
    end

    def calculate_probabilities(tokens)
      prob_numerator = {}
      v_size = vocab.size

      cat_prob = Math.log(1 / data.categories.count.to_f)
      total_example_count = data.total_examples.to_f

      data.each do |category|
        unless assume_uniform
          cat_prob = Math.log(data.example_count(category) / total_example_count)
        end

        log_probs = 0
        denominator = (data.token_count(category) + @k * v_size).to_f
        tokens.each do |token|
          numerator = data.count_of_token_in_category(category, token) + @k
          log_probs += Math.log( numerator / denominator )
        end
        prob_numerator[category] = log_probs + cat_prob
      end
      normalize(prob_numerator)
    end

    def normalize(prob_numerator)
      normalizer = 0
      prob_numerator.each {|cat, numerator| normalizer += numerator }
      intermed = {}
      renormalizer = 0
      prob_numerator.each do |cat, numerator|
        intermed[cat] = normalizer / numerator.to_f
        renormalizer += intermed[cat]
      end
      final_probs = {}
      intermed.each do |cat, value|
        final_probs[cat] = value / renormalizer.to_f
      end
      final_probs
    end
  end

  module Result
    def max_class
      keys.max{ |a,b| self[a] <=> self[b] }
    end
  end
end
