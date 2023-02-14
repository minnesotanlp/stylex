from tqdm import tqdm

class Trainer:
    def run_train(self, hp, model, optimizer, evaluator, train_loader, dev_loader,
                  writer=None, criterion='acc', early_stopping=False):
        prev_loss = float('inf')
        best_epoch = -1
        global_steps = 0
        best_dev_score = -1
        patience = 0
        for epoch in range(hp.train_epoch):
            model.train()

            total_loss = 0
            total_style_loss = 0
            total_perception_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            # Train the data for one epoch
            epoch_iterator = tqdm(train_loader, desc=f"Iteration {epoch}")

            for step, batch in enumerate(epoch_iterator):
                batch = tuple(t.to(hp.get_device()) for t in batch)
                # Unpack the inputs from our dataloader
                b_input_ids, b_input_mask, b_labels, b_perception_scores, b_perception_mask, b_perception_type = batch
                optimizer.zero_grad()

                losses, logits, outputs = model(b_input_ids, attention_mask=b_input_mask,
                                                style_label_ids=b_labels, perception_scores=b_perception_scores,
                                                perception_mask=b_perception_mask, perception_type=b_perception_type)
                loss = losses[2] if hp.perception_only else losses[0]
                
                if writer is not None:
                    writer.add_scalar("Loss all", loss, global_steps)
                if hp.get_perception_loss:
                    style_loss, perception_loss = losses[1:]
                    total_style_loss += style_loss
                    total_perception_loss += perception_loss

                    if writer is not None:
                        writer.add_scalar("Style loss", style_loss, global_steps)
                        writer.add_scalar("Word importance loss", perception_loss, global_steps)

                # Backward pass
                loss.backward()

                # Update parameters and take a step using the computed gradient
                optimizer.step()

                # Update tracking variables
                total_loss += loss.item()
                nb_tr_examples += b_input_ids.size(0)
                nb_tr_steps += 1
                global_steps += 1

            if hp.get_perception_loss:
                print(f"Train loss at epoch {epoch}: {total_loss / nb_tr_steps:.8f} \t "
                      f"Style loss: {total_style_loss / nb_tr_steps:.8f} \t "
                      f"Perception loss: {total_perception_loss / nb_tr_steps:.8f}")
            else:
                print(f"Train loss at epoch {epoch}: {total_loss / nb_tr_steps:.8f}")

            dev_accuracy, dev_perception_f1, dev_style_f1 = evaluator.eval(model, dev_loader)

            if writer is not None:
                writer.add_scalar("Dev Acc", dev_accuracy, epoch)

            if criterion == 'acc':
                dev_score = dev_accuracy
            elif criterion == 'perception_f1':
                dev_score = dev_perception_f1
            elif criterion == 'style_f1':
                dev_score = dev_style_f1
            else:
                raise RuntimeError('Invalid criterion!')

            if dev_score > best_dev_score:
                model.save_pretrained(hp.output_dir)
                print("Saving to {} at iteration: {} epoch: {}".format(hp.output_dir, nb_tr_steps, epoch))
                best_epoch = epoch
                best_dev_score = dev_score

            if early_stopping:
                if prev_loss > total_loss / nb_tr_steps:
                    prev_loss = total_loss / nb_tr_steps
                    patience = 0
                else:
                    patience += 1
                    if patience > hp.patience:
                        print("Stop training at epoch: {}".format(epoch))
                        print("Prev loss: ", prev_loss)
                        print("Current loss: ", total_loss / nb_tr_steps)
                        break

        print(f"Idx {hp.exp_idx} | Epoch with the highest {criterion} {best_dev_score:.5f}: {best_epoch}")
        return best_dev_score, best_epoch
